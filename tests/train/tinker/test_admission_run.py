"""Group admission and refill in the trainer's own loop (Tinker faked): RFC 0019's exit checks for its milestone M5, and
the risks the plan names.

Each group returns one datum whose first token is its row's position in the order, so the datums a step trains say
which positions it consumed; a group's verdict (``trained`` or ``uniform``) is the test's, by position or by a hash, and
its members agree with it, so ``informative`` rejects exactly the uniform ones.

- At ``max_staleness: 0`` a step starts a prefix of the order: its first ``batch_size`` positions and one more for each
  rejection, up to its cap; with a cap of 1.0 it starts and sends what all mode starts and sends.
- A step closes once it has counted its cap, a short step trains what it admitted (a partial last microbatch weighed
  by its share of groups), and a step that admitted nothing takes no update; the run ends after its planned steps.
- No group is stranded at ``k = 0`` or on the last step; markers of an earlier step pass over, so they never crowd out
  admitted groups; a rejected group's slot refills while the trainer trains.
- Groups that end at once (failed, or without a datum) keep their slots until a step dequeues them, so an outage that
  starts while the trainer trains does not start groups as fast as they end.
- Every position has exactly one fate, across interrupts and a crash (a crash, or a stop before a start's first step
  completes, loses the rejections made since the last full save, whose positions the resume draws again); the
  statistics count each fate once; a resume that cannot read the dispatcher's state goes on where the order stood.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import signal
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
import yaml
from test_dispatch_run import _crash_after, _Groups, _Recording, _run, _state
from tinker_service_fakes import FakeTinkerService, make_pruning_trainer
from trainer_fakes import make_trainer

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.task_stats import MAIN, MemberResult, read_events
from platoon.train.tinker.checkpoint_pruning import read_checkpoint_records
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.dispatcher import POSITION_COLUMN, Dispatcher
from platoon.train.tinker.task_records import CHANGED_KEY
from platoon.train.tinker.workflows.group_rollout_workflow import _TaskRolloutOutput
from platoon.utils.config import resolve_config
from platoon.utils.rollout_workload import RolloutWorkload


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


# --- the fakes --------------------------------------------------------------------------------------------------------


def _datum(position: int, version: int, advantage: float) -> tinker.Datum:
    """A datum whose first token is the position of the row its group ran, with two action tokens."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, 1, 2]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, advantage, advantage],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


def _member(passed: bool) -> MemberResult:
    return MemberResult(
        reward=float(passed),
        scored=float(passed),
        valid=True,
        outcome="scored",
        disposition="completed",
        reward_key=None,
        steps=1,
        output_tokens=2,
        seconds=None,
        versions={MAIN: (0, 0)},
    )


def by_position(*informative: int, period: int = 3):
    """A verdict function: ``trained`` at the positions whose remainder by ``period`` is one of ``informative``."""

    return lambda position, task_id: "trained" if position % period in informative else "uniform"


def by_hash(share: float, salt: str = ""):
    """A verdict function: ``trained`` for a fixed share of (task, position), by a hash of both."""

    def verdict(position: int, task_id: str) -> str:
        digest = hashlib.sha256(f"{salt}:{task_id}:{position}".encode()).digest()
        return "trained" if int.from_bytes(digest[:8], "big") / 2**64 < share else "uniform"

    return verdict


class _VerdictGroups:
    """A training workflow whose groups return a ``GroupRolloutWorkflow`` output: one datum of the row's position
    (advantage 1 when trained, 0 when uniform), the verdict ``verdict(position, task_id)`` gives, members that agree
    with it (one pass among four when trained, none when uniform), its label and entry.  Each takes ``delay(row)``
    seconds; ``on_start(row, count)`` runs as each starts; positions in ``fail`` raise."""

    def __init__(self, llm, *, verdict, delay=lambda row: 0.005, on_start=None, fail=(), column=ENVIRONMENT_COLUMN):
        self.llm = llm
        self.verdict = verdict
        self.delay = delay
        self.on_start = on_start
        positions = set() if callable(fail) else set(fail)
        self.fail = fail if callable(fail) else (lambda row: row[POSITION_COLUMN] in positions)
        self.column = column
        self.started: list[int] = []
        self.verdicts: dict[int, str] = {}

    async def arun_episode(self, data):
        position = data[POSITION_COLUMN]
        self.started.append(position)
        if self.on_start is not None:
            self.on_start(data, len(self.started))
        version = self.llm.version
        await asyncio.sleep(self.delay(data))
        if self.fail(data):
            raise RuntimeError(f"the group of position {position} fails")
        verdict = self.verdict(position, data["task_id"])
        self.verdicts[position] = verdict
        trained = verdict == "trained"
        label = data.get(self.column)
        return _TaskRolloutOutput(
            [_datum(position, version, 1.0 if trained else 0.0)],
            workload=RolloutWorkload(
                trajectories=4, postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1
            ),
            requested_rollouts=4,
            observed_rollouts=4,
            trainable_rollouts=1 if trained else 0,
            task_retained_datums=1,
            label=label,
            entry=label,
            members=tuple(_member(trained and index == 0) for index in range(4)),
            verdict=verdict,
        )


def _config(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]] | None = None, **extra: Any):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "train": {"model_name": "m", "renderer_name": "r", "workflow_config": {"group_size": 2}, **train},
        "eval": {"every": 0},
        "watchdog": {"enabled": False},
        "log_path": str(tmp_path / "logs"),
        **extra,
    }
    if environments is not None:
        data["environments"] = [{"rollout": "x:y", **entry} for entry in environments]
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


class _Steps(_Recording):
    """Records each forward/backward's datums: (first token, advantage of an action token)."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(delay)
        self.sent: list[list[tuple[int, float]]] = []

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.sent.append(
            [
                (datum.model_input.to_ints()[0], float(datum.loss_fn_inputs["advantages"].to_torch()[1]))
                for datum in datums
            ]
        )
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def _trainer(monkeypatch, config, *, tasks: int = 400, client: Any = None, rows: list[dict[str, Any]] | None = None):
    """``(trainer, logged stats per step, client)`` of a run on fakes."""

    client = client if client is not None else _Steps()
    trainer, _saves, _logged = make_trainer(monkeypatch, config, tasks=tasks, training_client=client)
    logged: list[dict[str, Any]] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(stats)
    )
    if rows is not None:
        from datasets import Dataset

        trainer.train_dataset = Dataset.from_list(rows)
    return trainer, logged, client


def _train(trainer, workflow, timeout: float = 120) -> None:
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), timeout))


def _events(trainer) -> list[dict[str, Any]]:
    return list(read_events(Path(trainer.run_log_path)))


def check_fates_once(trainer) -> dict[int, str]:
    """Every position has at most one fate in the event log, and the statistics' per-task fates are the events'; the
    fate of each position."""

    events = _events(trainer)
    position = {(e["start"], e["seq"]): e["position"] for e in events if e["event"] == "draw"}
    task = {(e["start"], e["seq"]): e["task_id"] for e in events if e["event"] == "draw"}
    fates = [
        (position[(e["start"], e["seq"])], task[(e["start"], e["seq"])], e["fate"])
        for e in events
        if e["event"] == "fate"
    ]
    by_position = Counter(position for position, _task, _fate in fates)
    assert all(count == 1 for count in by_position.values()), by_position.most_common(3)
    store = trainer._recorder.store
    counted: Counter[tuple[str, str]] = Counter((task_id, fate) for _position, task_id, fate in fates)
    for task_id in store.task_ids():
        for fate, count in store.task(task_id).fates.items():
            assert counted[(task_id, fate)] == count, (task_id, fate)
    return {position: fate for position, _task, fate in fates}


# --- the oracle at k = 0 ----------------------------------------------------------------------------------------------


def _starts(logged: list) -> Any:
    """An on_start hook that records which step each group started in: the steps logged when it started."""

    starts: dict[int, int] = {}

    def hook(row, count) -> None:
        starts[row[POSITION_COLUMN]] = len(logged)

    return starts, hook


def test_at_k_0_with_two_groups_in_three_uniform_every_step_admits_its_batch_at_three_draws_each(tmp_path, monkeypatch):
    """The design's own check: positions 2 mod 3 are informative, B = 8, cap 3.0 (C = 24), the groups completing in a
    seeded random order: each step starts 24 consecutive positions, admits exactly 8, and trains their datums."""

    config = _config(
        tmp_path,
        {
            "batch_size": 8,
            "max_training_steps": 4,
            "max_staleness": 0,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    starts, hook = _starts(logged)
    pace = random.Random(5)
    workflow = _VerdictGroups(
        trainer._model_info.llm, verdict=by_position(2), delay=lambda row: pace.uniform(0, 0.01), on_start=hook
    )
    _train(trainer, workflow)

    assert len(logged) == 4
    for step in range(4):
        started = sorted(position for position, at in starts.items() if at == step)
        assert started == list(range(24 * step, 24 * step + 24))
        assert sorted(token for token, _advantage in client.sent[step]) == list(range(24 * step + 2, 24 * step + 24, 3))
        stats = logged[step]
        assert stats["train/admission/groups_admitted"] == 8 and stats["train/admission/short"] == 0
        assert stats["train/admission/draws_per_admitted"] == pytest.approx(3.0)
    assert sorted(workflow.started) == list(range(96))
    fates = check_fates_once(trainer)
    assert Counter(fates.values()) == {"consumed": 32, "rejected": 64}


def _walk(verdicts, *, batch: int, cap: int, steps: int) -> list[tuple[list[int], int]]:
    """The positions each step of a k = 0 run under informative starts, and how many it admits, walking the order: a
    step takes positions until it has admitted ``batch`` or taken ``cap``."""

    walked, position = [], 0
    for _ in range(steps):
        taken, admitted = [], 0
        while admitted < batch and len(taken) < cap:
            taken.append(position)
            admitted += verdicts(position) == "trained"
            position += 1
        walked.append((taken, admitted))
    return walked


def test_at_k_0_each_step_starts_and_admits_what_a_walk_of_the_order_gives(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {
            "batch_size": 8,
            "max_training_steps": 5,
            "max_staleness": 0,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    starts, hook = _starts(logged)
    pace = random.Random(11)
    verdict = by_hash(0.3, "walk")
    workflow = _VerdictGroups(
        trainer._model_info.llm, verdict=verdict, delay=lambda row: pace.uniform(0, 0.01), on_start=hook
    )
    _train(trainer, workflow)

    tasks = {e["position"]: e["task_id"] for e in _events(trainer) if e["event"] == "draw"}
    walked = _walk(lambda position: verdict(position, tasks[position]), batch=8, cap=24, steps=5)
    for step, (taken, admitted) in enumerate(walked):
        assert sorted(position for position, at in starts.items() if at == step) == taken
        assert logged[step]["train/admission/groups_admitted"] == admitted
        assert logged[step]["train/admission/counted"] == len(taken)
    check_fates_once(trainer)


def test_a_cap_of_one_starts_and_sends_what_all_mode_starts_and_sends(tmp_path, monkeypatch):
    """refill.max_groups_per_step 1.0 (C = B) at k = 0: each step starts the batch all mode starts, and sends the
    datums all mode sends (the rejected groups are the uniform ones all mode filters out), each step's advantages
    scaled by one factor, the ratio of the two runs' normalization token counts."""

    runs = {}
    for mode in ("all", "informative"):
        config = _config(
            tmp_path / mode,
            {
                "batch_size": 8,
                "max_training_steps": 6,
                "max_staleness": 0,
                "max_concurrent_rollouts": 16,
                "group_admission": mode,
                "refill": {"max_groups_per_step": 1.0},
            },
        )
        trainer, logged, client = _trainer(monkeypatch, config)
        starts, hook = _starts(logged)
        pace = random.Random(3)
        workflow = _VerdictGroups(
            trainer._model_info.llm,
            verdict=by_hash(0.4, "cap1"),
            delay=lambda row: pace.uniform(0, 0.01),
            on_start=hook,
        )
        _train(trainer, workflow)
        runs[mode] = (starts, client, workflow)
    (all_starts, all_client, _), (rule_starts, rule_client, _) = runs["all"], runs["informative"]
    assert all_starts == rule_starts  # the same positions, in the same steps
    assert len(all_client.sent) == len(rule_client.sent)
    for all_step, rule_step in zip(all_client.sent, rule_client.sent):
        assert sorted(token for token, _ in all_step) == sorted(token for token, _ in rule_step)
        admitted = len(rule_step)
        ratios = [rule[1] / base[1] for base, rule in zip(sorted(all_step), sorted(rule_step))]
        # All mode normalizes by the batch's 8 groups, rule mode by its own admitted ones (float32 advantages).
        assert ratios == pytest.approx([8 / admitted] * admitted, rel=1e-6)


# --- short steps, empty steps, warnings -------------------------------------------------------------------------------


def test_a_run_whose_groups_are_all_uniform_counts_its_cap_each_step_trains_nothing_and_ends(
    tmp_path, monkeypatch, caplog
):
    """Every group uniform for the first five steps, then a mix: each of those steps counts exactly C at k = 0, admits
    nothing, takes no optimizer step, and the run finishes on time; one warning for the first short step, one for the
    first empty one."""

    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 8,
            "max_staleness": 0,
            "max_concurrent_rollouts": 8,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    starts, hook = _starts(logged)
    workflow = _VerdictGroups(
        trainer._model_info.llm,
        verdict=lambda position, task_id: "trained" if position >= 60 and position % 2 else "uniform",
        on_start=hook,
    )
    with caplog.at_level(logging.INFO):
        _train(trainer, workflow)

    assert len(logged) == 8
    for step in range(5):
        stats = logged[step]
        assert stats["train/admission/counted"] == 12 and stats["train/admission/groups_admitted"] == 0
        assert stats["train/admission/short"] == 1 and stats["train/batch/optimizer_updates"] == 0
        assert "train/admission/draws_per_admitted" not in stats
        assert sum(at == step for at in starts.values()) == 12  # no busy loop: the cap's worth of draws a step
    assert client.optim_steps == 3
    warned = "admitted no group: of the groups it dequeued, 12 were rejected, 0 stale and 0 failed, and it counted 12"
    assert caplog.text.count(warned) == 1  # warned of once
    assert "The policy may have solved the pool" in caplog.text
    assert "fewer than train.batch_size" not in caplog.text  # no step was short but empty
    reason = "it admitted no group (every one of the 12 groups it counted was rejected, stale or failed)"
    assert caplog.text.count(reason) == 5  # each step's log line says why it took no update


def test_a_step_that_keeps_waiting_warns_each_ten_batches_it_counts(tmp_path, monkeypatch, caplog):
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 1,
            "max_staleness": 0,
            "max_concurrent_rollouts": 4,
            "group_admission": "informative",
            "refill": {"max_groups_per_step": 25},
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=lambda position, task_id: "uniform", delay=lambda row: 0)
    with caplog.at_level(logging.WARNING):
        _train(trainer, workflow)
    assert logged[0]["train/admission/counted"] == 50
    waiting = [record.getMessage() for record in caplog.records if "waits for more" in record.getMessage()]
    assert len(waiting) == 2 and "has counted 20 groups" in waiting[0] and "has counted 40 groups" in waiting[1]


def test_a_step_made_short_by_failed_groups_says_so_and_points_at_them(tmp_path, monkeypatch, caplog):
    """Ten of the twelve groups the first step counts fail: its warning counts them and names the failed groups, not
    the rule's pass rate, as the cause."""

    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 2,
            "max_staleness": 0,
            "max_concurrent_rollouts": 8,
            "group_admission": "informative",
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config)
    workflow = _VerdictGroups(
        trainer._model_info.llm,
        verdict=lambda position, task_id: "trained",
        fail=lambda row: 2 <= row[POSITION_COLUMN] < 12,
    )
    with caplog.at_level(logging.WARNING):
        _train(trainer, workflow)
    assert logged[0]["train/admission/groups_admitted"] == 2 and logged[0]["train/admission/failed"] == 10
    [warning] = [
        record.getMessage() for record in caplog.records if "fewer than train.batch_size" in record.getMessage()
    ]
    assert "of the groups it dequeued, 0 were rejected, 0 stale and 10 failed" in warning
    assert "Most of them failed" in warning and "raise train.refill.max_groups_per_step" not in warning


@pytest.mark.parametrize(("cap", "admitted", "scaled"), [(1.5, 5, True), (1.0, 3, False)])
def test_a_short_step_s_partial_last_microbatch_weighs_by_its_share_of_groups(
    tmp_path, monkeypatch, cap, admitted, scaled
):
    """Two microbatches of 4 groups a step: a step that closes with 5 admitted sends a full microbatch and a partial
    one of 1 group, whose advantages are a quarter of what they would be unscaled; one that closes with 3 admitted
    sends one microbatch, unscaled."""

    config = _config(
        tmp_path,
        {
            "batch_size": 8,
            "num_microbatches": 2,
            "max_training_steps": 1,
            "max_staleness": 0,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
            "refill": {"max_groups_per_step": cap},
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    informative = set(range(admitted))  # the step's first positions, all within its cap
    workflow = _VerdictGroups(
        trainer._model_info.llm, verdict=lambda position, task_id: "trained" if position in informative else "uniform"
    )
    _train(trainer, workflow)

    assert logged[0]["train/admission/groups_admitted"] == admitted
    if scaled:
        assert [len(sent) for sent in client.sent] == [4, 1]
        assert [advantage for _token, advantage in client.sent[0]] == pytest.approx([1 / 8] * 4)  # 4 groups × 2 tokens
        assert client.sent[1][0][1] == pytest.approx(1 / 8)  # unscaled 1 / 2: a quarter of it
    else:
        assert [len(sent) for sent in client.sent] == [3]
        assert [advantage for _token, advantage in client.sent[0]] == pytest.approx([1 / 6] * 3)


def test_with_one_microbatch_a_short_step_s_gradient_is_the_mean_over_its_admitted_groups(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {
            "batch_size": 8,
            "max_training_steps": 1,
            "max_staleness": 0,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
            "refill": {"max_groups_per_step": 1.5},
        },
    )
    trainer, _logged, client = _trainer(monkeypatch, config)
    workflow = _VerdictGroups(
        trainer._model_info.llm, verdict=lambda position, task_id: "trained" if position < 5 else "uniform"
    )
    _train(trainer, workflow)
    assert [len(sent) for sent in client.sent] == [5]
    assert [advantage for _token, advantage in client.sent[0]] == pytest.approx([1 / 10] * 5)


# --- staleness, stranded groups, no bound -----------------------------------------------------------------------------


def test_at_k_0_no_group_is_stranded_and_every_group_a_step_starts_is_counted_by_it(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {
            "batch_size": 6,
            "max_training_steps": 6,
            "max_staleness": 0,
            "max_concurrent_rollouts": 12,
            "group_admission": "informative",
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config)
    starts, hook = _starts(logged)
    pace = random.Random(8)
    workflow = _VerdictGroups(
        trainer._model_info.llm, verdict=by_hash(0.25, "k0"), delay=lambda row: pace.uniform(0, 0.02), on_start=hook
    )
    _train(trainer, workflow)
    assert all(stats["train/batch/stale_groups"] == 0 for stats in logged)
    for step, stats in enumerate(logged):
        assert sum(at == step for at in starts.values()) == stats["train/admission/counted"]
    fates = check_fates_once(trainer)
    assert set(fates.values()) <= {"consumed", "rejected"} and len(fates) == len(workflow.started)
    final = _final_state(trainer)
    assert final["outstanding"] == []  # the last step started nothing it did not count


def _final_state(trainer) -> dict[str, Any]:
    run_dir = Path(trainer.run_log_path)
    [state] = sorted(run_dir.glob("tasks/dispatch-*-final.json"))
    return json.loads(state.read_text())


@pytest.mark.parametrize("staleness", [3, None])
def test_on_the_last_step_no_group_starts_that_no_step_counts(tmp_path, monkeypatch, staleness):
    train = {"batch_size": 4, "max_training_steps": 5, "max_concurrent_rollouts": 64, "group_admission": "informative"}
    if staleness is not None:
        train["max_staleness"] = staleness
    config = _config(tmp_path, train)
    trainer, logged, _client = _trainer(monkeypatch, config, client=_Steps(delay=0.02))
    starts, hook = _starts(logged)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=by_hash(0.3, "last"), on_start=hook)
    _train(trainer, workflow)
    assert len(logged) == 5
    # The groups the final save holds outstanding were started before the last step: none started during it is left.
    final = _final_state(trainer)
    started_last = {position for position, at in starts.items() if at == 4}
    assert not started_last & {position for _start, _seq, _task, position in final["outstanding"]}
    assert all(stats["train/admission/counted"] <= 12 for stats in logged)
    check_fates_once(trainer)


@pytest.mark.parametrize("k", [1, 3])
def test_stale_groups_count_toward_the_cap_not_the_batch_and_no_trained_group_is_older_than_its_bound(
    tmp_path, monkeypatch, k
):
    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 8,
            "max_staleness": k,
            "max_concurrent_rollouts": 64,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config, client=_Steps(delay=0.01))
    pace = random.Random(k)
    versions: dict[int, int] = {}
    llm = trainer._model_info.llm

    def hook(row, count) -> None:
        versions[row[POSITION_COLUMN]] = llm.version

    workflow = _VerdictGroups(
        llm, verdict=by_hash(0.5, "stale"), delay=lambda row: 0.3 if pace.random() < 0.1 else 0.002, on_start=hook
    )
    _train(trainer, workflow)
    stale = sum(stats["train/batch/stale_groups"] for stats in logged)
    assert stale == sum(stats["train/admission/stale"] for stats in logged)
    sent = iter(client.sent)
    for stats in logged:
        if not stats["train/batch/optimizer_updates"]:
            assert stats["train/admission/groups_admitted"] == 0
            continue
        trained = next(sent)
        assert len(trained) == stats["train/admission/groups_admitted"] <= 4
        now = stats["train/policy/version"] - 1  # the version the step trained at, before its update
        assert all(now - versions[token] <= k for token, _advantage in trained)  # the stale filter keeps the bound
    fates = check_fates_once(trainer)
    assert Counter(fates.values())["stale"] == stale


# --- groups that fail, and markers ------------------------------------------------------------------------------------


def test_groups_that_fail_at_once_at_k_1_burn_at_most_the_cap_a_step_and_the_run_ends(tmp_path, monkeypatch):
    """Every group fails at once for the first 100 positions (k = 1, eight groups at once): a failed group keeps its
    slot until a step dequeues its marker, so the dispatcher starts groups no faster than steps take them, and a step
    counts at most its cap: it starts at most its cap and the horizon's worth of groups, step after step."""

    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 8,
            "max_staleness": 1,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config)
    starts, hook = _starts(logged)
    workflow = _VerdictGroups(
        trainer._model_info.llm,
        verdict=by_hash(0.5, "fail"),
        delay=lambda row: 0.0,
        on_start=hook,
        fail=lambda row: row[POSITION_COLUMN] < 100,
    )
    _train(trainer, workflow)
    assert len(logged) == 8
    for step, stats in enumerate(logged):
        assert stats["train/admission/counted"] <= 12
        started = [position for position, at in starts.items() if at == step]
        # While every group fails: a slot frees only as a step dequeues a marker, so a step starts at most what it
        # dequeues, the earlier steps' markers (a horizon's worth) and its own (its cap's worth).
        assert len(started) <= (12 + 8 if max(started, default=0) < 100 else 12 + 2 * 8), (step, len(started))
    fates = check_fates_once(trainer)
    assert all(fates[position] == "failed" for position in range(100) if position in fates)
    assert sum(stats["train/admission/failed"] for stats in logged) >= 12


class _Outage(_VerdictGroups):
    """Positions from ``down`` on end at once: by raising (a failed group), or with an output that carries no datum,
    verdict ``no_results``, as a ``GroupRolloutWorkflow`` returns when every member failed."""

    def __init__(self, llm, *, down: int, how: str, **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self.down, self.how = down, how

    async def arun_episode(self, data):
        position = data[POSITION_COLUMN]
        if position < self.down:
            return await super().arun_episode(data)
        self.started.append(position)
        if self.on_start is not None:
            self.on_start(data, len(self.started))
        await asyncio.sleep(0)
        if self.how == "raise":
            raise RuntimeError(f"the environment is down (position {position})")
        return _TaskRolloutOutput(
            [],
            workload=RolloutWorkload(
                trajectories=4, postmerge_datums=0, policy_eligible_datums=0, post_sampling_datums=0
            ),
            requested_rollouts=4,
            observed_rollouts=0,
            trainable_rollouts=0,
            task_retained_datums=0,
            members=(),
            verdict="no_results",
        )


@pytest.mark.parametrize("how", ["raise", "no_results"])
@pytest.mark.parametrize("staleness", [1, None])
def test_an_outage_that_starts_while_the_trainer_trains_starts_groups_no_faster_than_steps_take_them(
    tmp_path, monkeypatch, how, staleness
):
    """Positions from 12 on end at once (B = 4, eight groups at once, 6 steps), the outage beginning while the trainer
    runs the forward/backward (0.5 s) of the steps that train the last good groups: a group that failed or carried no
    datum keeps its slot until a step dequeues its marker, as in all mode, so during each update the dispatcher starts
    at most the slots the horizon (or, without a bound, the end of the plan) leaves, where freeing those slots at once
    started groups as fast as they ended, thousands in an update."""

    train = {"batch_size": 4, "max_training_steps": 6, "max_concurrent_rollouts": 16, "group_admission": "informative"}
    if staleness is not None:
        train["max_staleness"] = staleness
    config = _config(tmp_path, train)
    client = _Steps(delay=0.5)
    trainer, logged, _ = _trainer(monkeypatch, config, tasks=50, client=client)
    training = {"now": False}
    during: list[int] = []
    forward_backward = client.forward_backward_async

    async def timed(datums, loss_fn, loss_fn_config):
        training["now"] = True
        try:
            return await forward_backward(datums, loss_fn, loss_fn_config)
        finally:
            training["now"] = False

    client.forward_backward_async = timed

    def hook(row, count) -> None:
        if training["now"]:
            during.append(row[POSITION_COLUMN])

    workflow = _Outage(
        trainer._model_info.llm,
        down=12,
        how=how,
        verdict=lambda position, task_id: "trained",
        delay=lambda row: 0.01,
        on_start=hook,
    )
    _train(trainer, workflow, timeout=300)
    assert len(logged) == 6
    updates = sum(stats["train/batch/optimizer_updates"] > 0 for stats in logged)
    assert 0 < updates <= 6
    # An update starts at most the slots of the horizon, (k + 1) × B, or without a bound of the plan's end, 6 × B,
    # less the step's admitted groups (at least one, to take an update) and the groups that hold theirs.
    room = (staleness + 1 if staleness is not None else 6) * 4 - 1
    assert len(during) <= updates * room, (len(during), updates)
    # Each step dequeues at most its cap's worth of its own markers, and those the horizon or the plan held for it.
    assert len(workflow.started) <= 6 * (12 + 6 * 4), len(workflow.started)
    check_fates_once(trainer)


def test_a_failed_group_is_decided_at_settle_frees_its_slot_and_is_never_kept(tmp_path, monkeypatch):
    service = FakeTinkerService()
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 4,
            "max_staleness": 2,
            "max_concurrent_rollouts": 48,
            "group_admission": "informative",
        },
        checkpoint={"recovery_every": 1},
    )
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=lambda position, task_id: "trained", fail=(1,))
    _run(trainer, workflow)
    run_dir = Path(trainer.run_log_path)
    events = _events(trainer)
    fate = [e for e in events if e["event"] == "fate" and e.get("admission") == "failed"]
    assert len(fate) == 1 and fate[0]["fate"] == "failed" and fate[0]["rule"] == "informative"
    for row in read_checkpoint_records(run_dir):
        state = json.loads((run_dir / row["dispatch"]["file"]).read_bytes())
        assert 1 not in [position for _start, _seq, _task, position in state["outstanding"]]
        kept = (state.get("kept") or {}).get("groups") or []
        assert (fate[0]["start"], fate[0]["seq"]) not in {tuple(key) for key in kept}
    check_fates_once(trainer)


# --- the rule's errors, groups without a verdict, evaluation ----------------------------------------------------------


class Raising:
    """A rule that raises on every third task, answers None on others, and an empty reason on the rest."""

    def __init__(self, setup, options=None) -> None:
        pass

    def admit(self, group, stats):
        number = group.seq % 3
        if number == 0:
            raise RuntimeError("a bug in the rule")
        if number == 1:
            return None
        from platoon.experimental import Admission

        return Admission(False, "")


def test_a_rule_that_fails_admits_the_group_logs_once_and_counts_every_failure(tmp_path, monkeypatch, caplog):
    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 3,
            "max_staleness": 1,
            "max_concurrent_rollouts": 16,
            "group_admission": f"{__name__}:Raising",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=lambda position, task_id: "trained")
    with caplog.at_level(logging.ERROR):
        _train(trainer, workflow)
    assert len(logged) == 3 and all(len(sent) == 4 for sent in client.sent)  # every group admitted, training went on
    assert caplog.text.count("failed on the group of task") == 1
    assert sum(stats["train/admission/errors"] for stats in logged) >= 12


class CountsChecks:
    """A rule whose load-time check records each call."""

    calls: list[str] = []

    def __init__(self, setup, options=None) -> None:
        pass

    @staticmethod
    def check(options, ctx) -> None:
        CountsChecks.calls.append(ctx.level)

    def admit(self, group, stats):
        return True


def test_the_trainer_s_start_runs_no_rule_s_check_again(tmp_path, monkeypatch):
    """platoon train runs each rule's check where it resolves the rules, with the run's workflow; the trainer's start
    resolves them again for its decisions without running the checks, which it could only run blind to the workflow."""

    from platoon.train.admissions import choose_admissions

    CountsChecks.calls = []
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 2,
            "max_staleness": 0,
            "max_concurrent_rollouts": 4,
            "group_admission": f"{__name__}:CountsChecks",
        },
    )
    choose_admissions(config)
    assert CountsChecks.calls == ["run"]
    trainer, logged, _client = _trainer(monkeypatch, config)
    _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=lambda position, task_id: "trained"))
    assert len(logged) == 2 and trainer._admissions is not None and CountsChecks.calls == ["run"]


def test_groups_without_a_verdict_are_admitted_as_unknown_and_train_as_all_mode_would(tmp_path, monkeypatch, caplog):
    """M3's workflow fake returns plain lists of datums: informative admits each group with reason unknown, warns once,
    and the run trains what all mode trains."""

    sent = {}
    for mode in ("all", "informative"):
        config = _config(
            tmp_path / mode,
            {
                "batch_size": 4,
                "max_training_steps": 4,
                "max_staleness": 0,
                "max_concurrent_rollouts": 8,
                "group_admission": mode,
            },
        )
        trainer, logged, client = _trainer(monkeypatch, config)
        with caplog.at_level(logging.WARNING):
            _train(trainer, _Groups(trainer._model_info.llm))
        sent[mode] = [sorted(token for token, _ in step) for step in client.sent]
        if mode == "informative":
            assert all(stats["train/admission/admitted/unknown"] == 4 for stats in logged)
    assert sent["all"] == sent["informative"]
    assert caplog.text.count("whose output carries no verdict") == 1


def test_evaluation_never_meets_a_rule(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 3,
            "max_staleness": 1,
            "max_concurrent_rollouts": 8,
            "group_admission": "informative",
        },
        eval={"every": 1},
    )
    from trainer_fakes import Workflow

    trainer, _saves, _logged = make_trainer(monkeypatch, config, eval_tasks=3)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=by_hash(0.5, "eval"))
    evaluation = Workflow(delay=0.0)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow, eval_workflow=evaluation), 120))
    settled = sum(1 for e in _events(trainer) if e["event"] == "settle")
    assert evaluation.started >= 3 and trainer._admissions.calls == settled


# --- metrics -------------------------------------------------------------------------------------------------------


def test_each_step_logs_what_it_admitted_rejected_and_dropped_by_label_and_reason(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 4,
            "max_staleness": 0,
            "max_concurrent_rollouts": 16,
            "group_admission": "informative",
        },
        environments=[{"name": "hard"}, {"name": "easy", "group_admission": "all"}],
    )
    rows = [
        {"task_id": f"{name}::{name}{index}", ENVIRONMENT_COLUMN: name}
        for index in range(40)
        for name in ("hard", "easy")
    ]
    trainer, logged, _client = _trainer(monkeypatch, config, rows=rows)
    workflow = _VerdictGroups(trainer._model_info.llm, verdict=by_hash(0.3, "metrics"))
    _train(trainer, workflow)
    totals = Counter()
    for stats in logged:
        hard = {key: value for key, value in stats.items() if key.startswith("train/admission/environment/hard/")}
        easy = {key: value for key, value in stats.items() if key.startswith("train/admission/environment/easy/")}
        assert easy["train/admission/environment/easy/rejected"] == 0  # all admits every group
        assert stats["train/admission/groups_admitted"] == (
            hard["train/admission/environment/hard/admitted"] + easy["train/admission/environment/easy/admitted"]
        )
        assert stats["train/admission/rejected"] == hard["train/admission/environment/hard/rejected"]
        assert stats["train/admission/rejected"] == stats.get("train/admission/rejected/uniform_fail", 0)
        totals.update(
            {"rejected": stats["train/admission/rejected"], "admitted": stats["train/admission/groups_admitted"]}
        )
        assert 0 <= stats["train/admission/informative_rate"] <= 1
        assert stats["train/admission/errors"] == 0
    assert totals["admitted"] == 16 and totals["rejected"] > 0


def test_all_mode_logs_no_admission_metric_and_builds_no_admission_set(tmp_path, monkeypatch):
    for written in (False, True):
        train = {"batch_size": 2, "max_training_steps": 2, "max_concurrent_rollouts": 8}
        if written:
            train["group_admission"] = "all"
        config = _config(
            tmp_path / str(written), train, environments=[{"name": "a", "group_admission": "all"}] if written else None
        )
        trainer, logged, _client = _trainer(monkeypatch, config)
        _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=by_hash(0.5, "all")))
        assert trainer._admissions is None
        assert not any(key.startswith("train/admission/") for stats in logged for key in stats)


# --- markers of an earlier step, and refill while the trainer trains --------------------------------------------------


def test_markers_of_an_earlier_step_pass_over_and_never_crowd_out_admitted_groups(tmp_path, monkeypatch):
    """Where the trainer is the bottleneck (groups of 5 ms, a forward/backward of 0.2 s), k = 1, 16 groups at once, one
    group in four admitted, 8 steps of 8: the markers of the groups rejected while a step trained pass over, so steps
    stay full; counting every marker (the design's counting, monkeypatched in) makes them short and lets admitted
    groups go stale.  The bounds are the planner's model's at these parameters (400 seeds): short steps and stale drops
    at most its 95th percentile under the plan's counting (2 and 7 a run), at least its 5th under the design's (6 and
    10)."""

    def run(design: bool, where: Path) -> tuple[int, int]:
        config = _config(
            where,
            {
                "batch_size": 8,
                "max_training_steps": 8,
                "max_staleness": 1,
                "max_concurrent_rollouts": 32,
                "group_admission": "informative",
            },
        )
        took = Dispatcher.took
        if design:

            def every_marker(self, kind, window=None, *, held=False):
                return took(self, kind, self.step if kind == "marker" else window, held=held)

            monkeypatch.setattr(Dispatcher, "took", every_marker)
        try:
            trainer, logged, _client = _trainer(monkeypatch, config, client=_Steps(delay=0.2))
            _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=by_position(0, period=4)))
        finally:
            monkeypatch.setattr(Dispatcher, "took", took)
        short = sum(stats["train/admission/short"] for stats in logged)
        stale = sum(stats["train/batch/stale_groups"] for stats in logged)
        return short, stale

    plan_short, plan_stale = run(False, tmp_path / "plan")
    design_short, design_stale = run(True, tmp_path / "design")
    assert plan_short <= 2 and plan_stale <= 7, (plan_short, plan_stale)
    assert design_short >= 6 and design_stale >= 10, (design_short, design_stale)


def test_a_rejection_refills_its_slot_while_the_trainer_trains(tmp_path, monkeypatch):
    """k = 1, permits for 2B groups, a forward/backward of 0.3 s, one group in four admitted: groups keep starting while
    the trainer trains in rule mode, where in all mode the horizon fills with settled groups and nothing runs."""

    started_during = {}
    for mode in ("all", "informative"):
        config = _config(
            tmp_path / mode,
            {
                "batch_size": 4,
                "max_training_steps": 4,
                "max_staleness": 1,
                "max_concurrent_rollouts": 16,
                "group_admission": mode,
            },
        )
        client = _Steps(delay=0.3)
        trainer, logged, _ = _trainer(monkeypatch, config, client=client)
        training = {"now": False}
        during: list[int] = []
        forward_backward = client.forward_backward_async

        async def timed(datums, loss_fn, loss_fn_config):
            training["now"] = True
            try:
                return await forward_backward(datums, loss_fn, loss_fn_config)
            finally:
                training["now"] = False

        client.forward_backward_async = timed

        def hook(row, count) -> None:
            if training["now"]:
                during.append(row[POSITION_COLUMN])

        workflow = _VerdictGroups(
            trainer._model_info.llm, verdict=by_position(0, period=4), delay=lambda row: 0.01, on_start=hook
        )
        _train(trainer, workflow)
        started_during[mode] = len(during)
    assert started_during["informative"] > started_during["all"] + 8, started_during


# --- exactly once ---------------------------------------------------------------------------------------------------


CAPPED = (
    {
        "batch_size": 4,
        "max_training_steps": 10,
        "max_staleness": 3,
        "max_concurrent_rollouts": 64,
        "rollout_group_limits": {"slow": 1},
        "group_admission": "informative",
    },
    [{"name": "quick"}, {"name": "slow"}],
)


def _capped_rows() -> list[dict[str, Any]]:
    """Twice as many quick tasks as slow ones, interleaved."""

    rows = []
    for index in range(30):
        rows += [{"task_id": f"quick::q{2 * index + offset}", ENVIRONMENT_COLUMN: "quick"} for offset in (0, 1)]
        rows.append({"task_id": f"slow::s{index}", ENVIRONMENT_COLUMN: "slow"})
    return rows


@pytest.mark.parametrize("interrupts", [0, 1, 2])
def test_a_run_with_rejections_interrupted_consumes_every_position_once(tmp_path, monkeypatch, interrupts):
    service = FakeTinkerService()
    train, environments = CAPPED
    config = _config(tmp_path, train, environments, checkpoint={"every": 100})
    admitted = 0
    for interrupt in [3, 3][:interrupts] + [None]:
        trainer, logged = make_pruning_trainer(monkeypatch, config, service)
        from datasets import Dataset

        trainer.train_dataset = Dataset.from_list(_capped_rows())

        def hook(row, started, logged=logged, interrupt=interrupt) -> None:
            # Once the start has committed some steps, so that its interrupt checkpoint keeps what it decided (an
            # interrupt in a start's first step saves none, and its decisions are lost as a crash's are).
            if interrupt is not None and len(logged) == interrupt and not getattr(hook, "sent", False):
                hook.sent = True
                signal.raise_signal(signal.SIGINT)

        workflow = _VerdictGroups(
            trainer._model_info.llm,
            verdict=by_hash(0.4, "once"),
            delay=lambda row: 0.04 if row[ENVIRONMENT_COLUMN] == "slow" else 0.003,
            on_start=hook,
        )
        _run(trainer, workflow)
        admitted += sum(stats["train/admission/groups_admitted"] for _step, stats in logged)
    run_dir = Path(trainer.run_log_path)
    names = [row["name"] for row in read_checkpoint_records(run_dir)]
    assert sum(name.endswith("_interrupted") for name in names) == interrupts
    events = list(read_events(run_dir))
    draws = {(e["start"], e["seq"]): e for e in events if e["event"] == "draw"}
    fates = [(draws[(e["start"], e["seq"])]["position"], e["fate"]) for e in events if e["event"] == "fate"]
    by_position = Counter(position for position, _fate in fates)
    assert all(count == 1 for count in by_position.values())
    assert {fate for _position, fate in fates} <= {"consumed", "stale", "failed", "skipped", "rejected"}
    # A rejection is final: no later start draws its position again (a draw a resume drew again may be rejected).
    rejected_at: dict[int, int] = {}
    for index, event in enumerate(events):
        if event["event"] == "fate" and event["fate"] == "rejected":
            rejected_at[draws[(event["start"], event["seq"])]["position"]] = index
        elif event["event"] == "draw" and event.get("reason") == "reissue":
            assert event["position"] not in rejected_at, event
    final = json.loads((run_dir / read_checkpoint_records(run_dir)[-1]["dispatch"]["file"]).read_bytes())
    outstanding = [position for _start, _seq, _task, position in final["outstanding"]]
    waiting = sorted(by_position) + outstanding
    assert sorted(waiting) == list(range(final["next_position"]))  # every position: one fate, or still outstanding
    consumed = [position for position, fate in fates if fate == "consumed"]
    assert len(consumed) == admitted <= 40  # what the committed steps admitted, each once
    snapshot = json.loads((run_dir / read_checkpoint_records(run_dir)[-1]["tasks"]["file"]).read_bytes())
    snapshot_fates = Counter()
    for task in snapshot["tasks"].values():
        snapshot_fates.update(task["fates"])
    assert snapshot_fates == Counter(fate for _position, fate in fates)


def test_a_crash_between_a_rejection_and_the_next_save_reissues_its_draw_once(tmp_path, monkeypatch):
    service = FakeTinkerService()
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 4,
            "max_staleness": 2,
            "max_concurrent_rollouts": 48,
            "group_admission": "informative",
        },
        checkpoint={"recovery_every": 1},
    )
    first, _logged = make_pruning_trainer(monkeypatch, config, service)
    slow_odd = lambda row: 0.08 if row[POSITION_COLUMN] % 2 else 0.005  # noqa: E731
    _run(first, _VerdictGroups(first._model_info.llm, verdict=by_position(0, period=2), delay=slow_odd))
    run_dir = Path(first.run_log_path)
    events = list(read_events(run_dir))
    rows = _crash_after(run_dir, "recovery-000001")
    state = _state(run_dir, rows[-1])
    outstanding = {(start, seq) for start, seq, _task, _position in state["outstanding"]}
    rejected_after = [
        e for e in events if e["event"] == "fate" and e["fate"] == "rejected" and (e["start"], e["seq"]) in outstanding
    ]
    assert rejected_after, "a group rejected after the checkpoint, which the crash loses"
    second, _logged = make_pruning_trainer(monkeypatch, config, service)
    _run(second, _VerdictGroups(second._model_info.llm, verdict=by_position(0, period=2), delay=slow_odd))
    events = list(read_events(run_dir))
    lost_keys = {(e["start"], e["seq"]) for e in rejected_after}
    reissued = [e for e in events if e["event"] == "draw" and tuple(e.get("reissues") or ()) in lost_keys]
    assert len(reissued) == len(lost_keys)  # each drawn again once
    # The event log keeps the crashed start's rejection beside the resumed start's fate of the same position, but the
    # statistics the run goes on with count each position once.
    position = {(e["start"], e["seq"]): e["position"] for e in events if e["event"] == "draw"}
    with_fate = {position[(e["start"], e["seq"])] for e in events if e["event"] == "fate"}
    snapshot = json.loads((run_dir / read_checkpoint_records(run_dir)[-1]["tasks"]["file"]).read_bytes())
    assert sum(sum(task["fates"].values()) for task in snapshot["tasks"].values()) == len(with_fate)


def test_a_resume_that_cannot_read_the_dispatcher_s_state_goes_on_where_the_order_stood(tmp_path, monkeypatch):
    """recovery-000004's state of the draws cannot be written, and the run dies right after its row: the resume counts
    the draws it left outstanding as lost and draws on from where the order stood, which the row records under a rule
    (a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch lags behind it), so that no
    position the run trained or rejected is drawn again."""

    from platoon.train.tinker import dispatcher as dispatcher_module
    from platoon.train.tinker import rl as rl_module

    service = FakeTinkerService()
    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 8,
            "max_staleness": 1,
            "max_concurrent_rollouts": 48,
            "group_admission": "informative",
        },
        checkpoint={"recovery_every": 1},
    )
    first, _logged = make_pruning_trainer(monkeypatch, config, service, tasks=40)
    run_dir = Path(first.run_log_path)
    write, checkpoint_state = dispatcher_module.write_tasks_file, rl_module.PlatoonTinkerRLTrainer._checkpoint_state
    captured: dict[str, int] = {}

    def refused(directory, name, content):
        if name.startswith("dispatch-") and "recovery-000004" in name:
            captured["next_position"] = json.loads(content)["next_position"]
            raise OSError(28, "No space left on device")
        return write(directory, name, content)

    async def capture(self, checkpoint, loop_state):
        if checkpoint == "recovery-000004":
            captured["events"] = len(list(read_events(run_dir)))
        return await checkpoint_state(self, checkpoint, loop_state)

    monkeypatch.setattr(dispatcher_module, "write_tasks_file", refused)
    monkeypatch.setattr(rl_module.PlatoonTinkerRLTrainer, "_checkpoint_state", capture)
    verdict = by_hash(0.3, "degraded")
    _run(first, _VerdictGroups(first._model_info.llm, verdict=verdict))
    monkeypatch.setattr(dispatcher_module, "write_tasks_file", write)
    monkeypatch.setattr(rl_module.PlatoonTinkerRLTrainer, "_checkpoint_state", checkpoint_state)
    rows = _crash_after(run_dir, "recovery-000004")
    assert "dispatch" not in rows[-1] and rows[-1]["next_position"] == captured["next_position"] > 4 * 2
    before = len(list(read_events(run_dir)))
    second, _logged = make_pruning_trainer(monkeypatch, config, service, tasks=40)
    _run(second, _VerdictGroups(second._model_info.llm, verdict=verdict))
    events = list(read_events(run_dir))
    position = {(e["start"], e["seq"]): e["position"] for e in events if e["event"] == "draw"}
    lineage = events[: captured["events"]] + events[before:]  # what the run went through, the crash's loss aside
    consumed = Counter(
        position[(e["start"], e["seq"])] for e in lineage if e["event"] == "fate" and e["fate"] != "lost"
    )
    assert consumed and [p for p, count in consumed.items() if count > 1] == []
    assert min(e["position"] for e in events[before:] if e["event"] == "draw") == captured["next_position"]


# --- resumes that change admission ------------------------------------------------------------------------------------


def test_a_resume_that_turns_admission_on_names_each_change_and_trains_the_kept_groups_once(
    tmp_path, monkeypatch, caplog
):
    service = FakeTinkerService()
    base = {"batch_size": 2, "max_training_steps": 4, "max_staleness": 2, "max_concurrent_rollouts": 48}
    before = _config(tmp_path, base, environments=[{"name": "a"}], checkpoint={"recovery_every": 1})
    first, _logged = make_pruning_trainer(monkeypatch, before, service)
    _run(first, _VerdictGroups(first._model_info.llm, verdict=by_position(0, period=2)))
    run_dir = Path(first.run_log_path)
    rows = _crash_after(run_dir, "recovery-000001")
    kept = (_state(run_dir, rows[-1]).get("kept") or {}).get("groups") or []
    (run_dir / "lock.json").write_text("{}\n")
    after = _config(
        tmp_path,
        {**base, "refill": {"max_groups_per_step": 4.0}},
        environments=[{"name": "a", "group_admission": "informative", "group_admission_options": {"band": [0.2, 0.8]}}],
        checkpoint={"recovery_every": 1},
    )
    second, logged = make_pruning_trainer(monkeypatch, after, service)
    with caplog.at_level(logging.WARNING):
        _run(second, _VerdictGroups(second._model_info.llm, verdict=by_position(0, period=2)))
    [changed] = json.loads((run_dir / "lock.json").read_text())[CHANGED_KEY]
    assert "environments[a].group_admission: all -> informative" in changed["changes"]
    # Turned on: the rule's change alone, the options and the cap now in force not set against all mode's.
    assert not [change for change in changed["changes"] if "options" in change or "max_groups_per_step" in change]
    events = list(read_events(run_dir))
    begun = next(index for index, e in enumerate(events) if e["event"] == "start" and e["start"] == 2)
    resumed = events[begun:]  # the crashed start's own events stay in the log before it
    consumed = Counter((e["start"], e["seq"]) for e in resumed if e["event"] == "fate" and e["fate"] == "consumed")
    assert kept and all(consumed[tuple(key)] == 1 for key in kept)  # each kept group trained once, not decided again


def test_a_resume_that_turns_admission_off_goes_back_to_all_mode(tmp_path, monkeypatch, caplog):
    service = FakeTinkerService()
    base = {"batch_size": 2, "max_training_steps": 4, "max_staleness": 2, "max_concurrent_rollouts": 48}
    first, _ = make_pruning_trainer(
        monkeypatch,
        _config(tmp_path, {**base, "group_admission": "informative"}, checkpoint={"recovery_every": 1}),
        service,
    )
    _run(first, _VerdictGroups(first._model_info.llm, verdict=by_position(0, period=2)))
    run_dir = Path(first.run_log_path)
    _crash_after(run_dir, "recovery-000001")
    (run_dir / "lock.json").write_text("{}\n")
    second, logged = make_pruning_trainer(
        monkeypatch, _config(tmp_path, base, checkpoint={"recovery_every": 1}), service
    )
    _run(second, _VerdictGroups(second._model_info.llm, verdict=by_position(0, period=2)))
    [changed] = json.loads((run_dir / "lock.json").read_text())[CHANGED_KEY]
    assert "environments[default].group_admission: informative -> all" in changed["changes"]
    assert second._admissions is None
    assert not any(key.startswith("train/admission/") for _step, stats in logged for key in stats)


# --- the rule's view --------------------------------------------------------------------------------------------------


class ReadsTheView:
    """Records, at each decision, the groups the view's store holds and the task's fates."""

    seen: list[tuple[int, int]] = []

    def __init__(self, setup, options=None) -> None:
        pass

    def admit(self, group, stats):
        groups = sum(stats.counts(task).groups for task in stats.task_ids())
        ReadsTheView.seen.append((group.seq, groups))
        return True


def test_a_rule_s_view_takes_no_settled_group_in(tmp_path, monkeypatch):
    """A decision reads the statistics as of the last flush: the groups settled since then are held, in draw order, so
    the store's count of groups moves only at the flush points (each step's metrics), never at a settle."""

    ReadsTheView.seen = []
    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 3,
            "max_staleness": 0,
            "max_concurrent_rollouts": 8,
            "group_admission": f"{__name__}:ReadsTheView",
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config)
    _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=by_hash(0.5, "view")))
    by_step = [{groups for seq, groups in ReadsTheView.seen if 4 * step <= seq < 4 * step + 4} for step in range(3)]
    assert by_step == [{0}, {4}, {8}]  # within a step, the store held the earlier steps' groups only


def test_a_rule_still_decides_when_the_statistics_stopped(tmp_path, monkeypatch):
    from platoon.train.task_stats import TaskStatsStore

    def settle(self, record):
        raise RuntimeError("a bug in the statistics")

    monkeypatch.setattr(TaskStatsStore, "settle", settle)
    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 3,
            "max_staleness": 0,
            "max_concurrent_rollouts": 8,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=by_position(0, period=2)))
    assert trainer._recorder.stopped
    assert all(stats["train/admission/rejected"] >= 1 for stats in logged[1:])  # informative still judged them
    assert all(token % 2 == 0 for sent in client.sent for token, _ in sent)


def test_group_record_that_raises_admits_the_group_and_nothing_escapes(tmp_path, monkeypatch):
    from platoon.train.tinker import task_records

    def broken(**kwargs):
        raise RuntimeError("cannot build a record")

    config = _config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 2,
            "max_staleness": 0,
            "max_concurrent_rollouts": 4,
            "group_admission": "informative",
        },
    )
    trainer, logged, client = _trainer(monkeypatch, config)
    trainer._begin_task_records = lambda resume_info, start_batch: setattr(trainer, "_recorder", None)
    monkeypatch.setattr(task_records, "group_record", broken)
    _train(trainer, _VerdictGroups(trainer._model_info.llm, verdict=lambda position, task_id: "uniform"))
    assert len(logged) == 2 and all(stats["train/admission/errors"] == 2 for stats in logged)


def test_the_rule_s_view_never_takes_the_held_records_in():
    """``SamplerSet.view(now, flush=False)``, the view a rule reads, leaves the settled groups' records held for the
    next flush; the view an adaptive decision reads takes them in."""

    import tempfile

    from datasets import Dataset
    from trainer_fakes import trainer_config

    from platoon.train.sampler_runtime import SamplerSet, TaskPool
    from platoon.train.tinker.dispatcher import SEQ_COLUMN
    from platoon.train.tinker.task_records import TaskRecorder

    with tempfile.TemporaryDirectory() as directory:
        config = trainer_config(Path(directory))
        rows = [{"task_id": f"train:{index}"} for index in range(4)]
        recorder = TaskRecorder(run_dir=directory, config=config, tasks=rows, start=1)
        recorder.draw(rows[0], seq=0, position=0, version=0)
        record = recorder.settle({**rows[0], SEQ_COLUMN: 0, POSITION_COLUMN: 0}, entry=None, label=None, output=[])
        assert record is not None and recorder._held == [record]
        pool = TaskPool.from_dataset(Dataset.from_list(rows), entries=("default",), column=ENVIRONMENT_COLUMN)
        samplers = SamplerSet.of_order(lambda position: rows[position], pool=pool)
        samplers.attach(recorder)
        quiet = samplers.view(0, flush=False)
        assert recorder._held == [record] and quiet.counts("train:0").groups == 0
        flushed = samplers.view(0)
        assert recorder._held == [] and flushed.counts("train:0").groups == 1
        recorder.close()


def test_per_entry_rules_train_an_entry_at_all_whole_and_shift_the_trained_mix(tmp_path, monkeypatch):
    """hard (informative, 60 tasks, a fifth of its groups informative) beside easy (all, 20 tasks, a fifth of its
    groups failing): easy's uniform groups train, its failed groups count toward the cap and not toward the batch, and
    the trained mix shifts from the draws' 3 to 1 to about 0.75 × 0.2 against 0.25 × 0.8 (a share of 43% for hard)."""

    config = _config(
        tmp_path,
        {"batch_size": 8, "max_training_steps": 12, "max_staleness": 0, "max_concurrent_rollouts": 32},
        environments=[{"name": "hard", "group_admission": "informative"}, {"name": "easy"}],
    )
    rows = [{"task_id": f"hard::h{index}", ENVIRONMENT_COLUMN: "hard"} for index in range(60)]
    rows += [{"task_id": f"easy::e{index}", ENVIRONMENT_COLUMN: "easy"} for index in range(20)]
    trainer, logged, client = _trainer(monkeypatch, config, rows=rows)
    hard = by_hash(0.2, "per-entry")

    def verdict(position: int, task_id: str) -> str:
        return hard(position, task_id) if task_id.startswith("hard::") else "uniform"

    workflow = _VerdictGroups(
        trainer._model_info.llm,
        verdict=verdict,
        fail=lambda row: row["task_id"].startswith("easy::") and row[POSITION_COLUMN] % 5 == 0,
    )
    _train(trainer, workflow)
    hard_admitted = sum(stats["train/admission/environment/hard/admitted"] for stats in logged)
    easy_admitted = sum(stats["train/admission/environment/easy/admitted"] for stats in logged)
    easy_failed = sum(stats["train/admission/environment/easy/failed"] for stats in logged)
    assert easy_failed > 0 and all(stats["train/admission/environment/easy/rejected"] == 0 for stats in logged)
    assert all(stats["train/admission/groups_admitted"] <= 8 for stats in logged)
    fates = check_fates_once(trainer)
    tasks = {e["position"]: e["task_id"] for e in _events(trainer) if e["event"] == "draw"}
    easy_consumed = [p for p, fate in fates.items() if fate == "consumed" and tasks[p].startswith("easy::")]
    assert easy_admitted == len(easy_consumed) > 0  # easy's uniform groups trained, as under all
    share = hard_admitted / (hard_admitted + easy_admitted)
    assert 0.43 - 0.15 <= share <= 0.43 + 0.15, share


def test_a_kept_failed_group_of_an_all_mode_checkpoint_is_taken_as_failed_under_a_rule(tmp_path, monkeypatch):
    """M3's run: at recovery-000001 the failed group of position 3 had settled and no step had consumed it. The resume,
    now under informative, consumes it as failed (counted toward the cap, not the batch), without rolling its task out
    again."""

    from test_dispatch_run import Q4_RUN, _Failing, _pruning_trainer

    service = FakeTinkerService()
    first, _logged, _client = _pruning_trainer(monkeypatch, tmp_path, service, *Q4_RUN)
    _run(first, _Failing(first._model_info.llm, [3]))
    run_dir = Path(first.run_log_path)
    rows = _crash_after(run_dir, "recovery-000001")
    [key] = [(start, seq) for start, seq, _task, position in _state(run_dir, rows[-1])["outstanding"] if position == 3]
    assert list(key) in (_state(run_dir, rows[-1]).get("kept") or {}).get("groups", [])
    (run_dir / "lock.json").write_text("{}\n")
    second, logged, _client = _pruning_trainer(
        monkeypatch, tmp_path, service, *Q4_RUN, "train.group_admission=informative"
    )
    workflow = _Groups(second._model_info.llm)
    _run(second, workflow)
    assert 3 not in workflow.started  # not rolled out again
    assert sum(stats["train/admission/failed"] for _step, stats in logged) == 1
    events = list(read_events(run_dir))
    begun = next(index for index, e in enumerate(events) if e["event"] == "start" and e["start"] == 2)
    fates = [e for e in events[begun:] if e["event"] == "fate" and (e["start"], e["seq"]) == key]
    assert [e["fate"] for e in fates] == ["failed"]


def test_a_run_platoon_0_3_0_interrupted_resumes_under_informative_naming_each_entry_s_rule(tmp_path, monkeypatch):
    """The 0.3.0 fixture of a weighted mixture with a capped slow entry, resumed under informative: the change is named
    for each entry, the groups the checkpoint kept train once, counted toward the batch, and are not rolled out
    again (M3's plain lists carry no verdict: informative admits every group, as unknown)."""

    import shutil

    import resume_0_3_0

    name = "weighted-capped"
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "resume-0.3.0" / name
    run = dict(resume_0_3_0.RUNS[name])
    monkeypatch.setitem(resume_0_3_0.RUNS, name, {**run, "train": {**run["train"], "group_admission": "informative"}})
    trainer, workflow, steps = resume_0_3_0.start(monkeypatch, name, tmp_path)
    run_dir = Path(trainer.run_log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture / "checkpoints.jsonl", run_dir / "checkpoints.jsonl")
    shutil.copytree(fixture / "tasks", run_dir / "tasks")
    (run_dir / "lock.json").write_text("{}\n")  # platoon train writes it before the trainer starts
    saved = json.loads(
        (
            run_dir
            / [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()][-1]["dispatch"][
                "file"
            ]
        ).read_bytes()
    )
    position_of = {(start, seq): position for start, seq, _task, position in saved["outstanding"]}
    kept = {position_of[tuple(key)] for key in saved["kept"]["groups"]}

    resume_0_3_0.run(trainer, workflow)

    [changed] = json.loads((run_dir / "lock.json").read_text())[CHANGED_KEY]
    assert "environments[quick].group_admission: all -> informative" in changed["changes"]
    assert "environments[slow].group_admission: all -> informative" in changed["changes"]
    trained = [position for step in steps for position in step]
    assert kept <= set(trained) and not kept & set(workflow.started)


def test_the_step_s_counts_are_the_dispatcher_s_after_every_item(tmp_path, monkeypatch):
    """At k = 1 with rejections, failures and stale groups: after every item a step dequeues, the dispatcher's counts
    move as the step's do, and at its close the step's counted and admitted groups (its metrics) are the dispatcher's;
    the queued markers of the current step never go negative."""

    config = _config(
        tmp_path,
        {
            "batch_size": 4,
            "max_training_steps": 8,
            "max_staleness": 1,
            "max_concurrent_rollouts": 32,
            "group_admission": "informative",
        },
    )
    trainer, logged, _client = _trainer(monkeypatch, config, client=_Steps(delay=0.01))
    closed: list[tuple[int, int]] = []
    seen: list[tuple[int, int, int]] = []
    took, step_started = Dispatcher.took, Dispatcher.step_started

    def watched_took(self, kind, window=None, *, held=False):
        counts = took(self, kind, window, held=held)
        seen.append((self.counted, self.admitted, self._markers_now))
        return counts

    def watched_start(self):
        closed.append((self.counted, self.admitted))
        step_started(self)

    monkeypatch.setattr(Dispatcher, "took", watched_took)
    monkeypatch.setattr(Dispatcher, "step_started", watched_start)
    pace = random.Random(2)
    workflow = _VerdictGroups(
        trainer._model_info.llm,
        verdict=by_hash(0.4, "counts"),
        delay=lambda row: 0.15 if pace.random() < 0.1 else 0.002,
        fail=lambda row: row[POSITION_COLUMN] % 11 == 5,
    )
    _train(trainer, workflow)
    assert len(closed) == len(logged) == 8
    for (counted, admitted), stats in zip(closed, logged):
        assert (counted, admitted) == (stats["train/admission/counted"], stats["train/admission/groups_admitted"])
    assert all(markers >= 0 for _counted, _admitted, markers in seen)
    check_fates_once(trainer)
