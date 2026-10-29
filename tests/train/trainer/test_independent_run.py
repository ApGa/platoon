"""Two policies that step independently (``train.stepping: independent``), whole starts on CPU on two fake Tinker
services (``fakes.independent``): main steps on 4 shares and the worker on 1, each on its own clock (stepping 1:4),
publishing to the rollout workers and saving its own rows on its own cadence, each full save a commit; every share
trains once in each policy's weights, and every group counts toward the plan, one without main's rows too; a resume goes
on from the newest commit; an interrupt with one policy's step open lets that step finish and saves both, also when the
step takes two updates and the stopped policy's horizon holds back the shares it still needs (it closes short), and a
stop in main's last step, with the worker behind, resumes to a final row of both; each step logs a row of its own, on
its policy's axis; a step hook updates the policy whose step ends, and is refused another, through its end or the run's
learner, and no commit saves a hook's state part-way through another policy's step end, and a hook whose state follows
each policy's own step resumes with it across a kill, also keeping only what a commit can name; a trainer of its own
steps each policy through the share ledger, stops on a stop request, and judges a share a resume put back at the age it
had; and a start keeps the stepping its run's checkpoints were saved with, and the rules of the stepping its trainer's
needs choose."""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, ClassVar

import pytest
from fakes.independent import (
    WORKER_URL,
    Lineages,
    Services,
    TaggedGroups,
    final_lineage,
    resumed_from,
    rows,
    run_dir,
    shares_put_back,
    start,
    tagged_row,
)

import platoon.experimental as px
from platoon.registry import register_component
from platoon.testing import trainers as checks
from platoon.train.checkpoints import commits as commits_module
from platoon.train.policy.spec import MAIN
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import OutputReport
from platoon.utils.config import ConfigError
from platoon.utils.stats_logger import StatsLogger


@pytest.fixture
def logged(monkeypatch) -> list[tuple[dict[str, Any], str]]:
    """Every row the run logged without a W&B step of its own, with its axis, in order."""

    found: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(StatsLogger, "log_row", lambda self, stats, *, axis: found.append((dict(stats), axis)))
    return found


def _once(trained: list[int]) -> bool:
    return all(count == 1 for count in Counter(trained).values())


def test_main_steps_on_4_shares_and_the_worker_on_1_each_on_its_own_clock(tmp_path, logged):
    lineages = Lineages()
    services, workflow = start(tmp_path, lineages, steps=3)

    [main], [worker] = services.main.training_clients, services.worker.training_clients
    assert (main.optimizer_steps, worker.optimizer_steps) == (3, 12)  # 12 groups: 3 steps of 4, 12 steps of 1
    # Every share trained once, in each policy's weights: every group's.
    run = run_dir(tmp_path)
    assert sorted(final_lineage(lineages, run)) == sorted(workflow.tags)
    assert sorted(final_lineage(lineages, run / "policies" / "worker")) == sorted(workflow.tags)

    # Each policy saves on its own cadence: main a commit at each of its steps, the worker at every second of its own.
    leader, own = rows(run), rows(run / "policies" / "worker")
    assert [row["name"] for row in leader] == [
        "recovery-000000",
        "recovery-000001",
        "recovery-000002",
        "recovery-000003",
        "final",
    ]
    commits = [row for row in own if "commit_seq" in row]
    assert [row["name"] for row in commits] == [f"recovery-worker-{step:06d}" for step in (2, 4, 6, 8, 10, 12)]
    assert [row["policies"]["worker"]["step"] for row in commits] == [2, 4, 6, 8, 10, 12]
    # A commit names every policy's row: a worker's, main's newest durable row, and the step main resumes at.
    for row in commits:
        named = row["policies"][MAIN]
        assert row["batch"] == named["step"] and any(each["state_path"] == named["state_path"] for each in leader)
    seqs = [row["commit_seq"] for row in leader + commits]
    assert len(set(seqs)) == len(seqs)
    assert leader[-1]["policies"]["worker"]["step"] == 12 and leader[-1]["policies"][MAIN]["step"] == 3

    # Each policy publishes after each of its own steps, to the rollout workers too.
    assert [version for version, policy in workflow.bindings if policy is None] == [0, 1, 2, 3]
    assert [version for version, policy in workflow.bindings if policy == "worker"] == list(range(13))

    # Each step's row: main's with today's names and the run's metrics, on the run's axis; the worker's under
    # policies/worker/, on its own.
    mains = [stats for stats, axis in logged if axis == "run/step"]
    workers = [stats for stats, axis in logged if axis == "policies/worker/step"]
    assert [stats["run/step"] for stats in mains] == [1, 2, 3]
    assert [stats["policies/worker/step"] for stats in workers] == list(range(1, 13))
    assert all(key.startswith("policies/worker/") for stats in workers for key in stats)
    assert all(stats["policies/worker/train/batch/optimizer_updates"] == 1.0 for stats in workers)
    assert [stats["train/batch/optimizer_updates"] for stats in mains] == [1.0, 1.0, 1.0]
    for key in (
        "train/dispatch/blocked_share/horizon:main",
        "train/dispatch/blocked_share/horizon:worker",
        "run/retained_groups",
        "policies/main/share_wait_s",
        "policies/main/retrained_shares",
        "policies/main/peer_lag/worker",
    ):
        assert key in mains[0], key
    assert {"policies/worker/share_wait_s", "policies/worker/peer_lag/main"} <= set(workers[0])


def test_a_resume_goes_on_from_the_newest_commit_each_policy_from_the_row_it_names(tmp_path):
    lineages = Lineages()
    _, first = start(tmp_path, lineages, steps=2)
    again, second = start(tmp_path, lineages, steps=3)

    run = run_dir(tmp_path)
    final = rows(run)[-1]
    assert [client.optimizer_steps for client in again.main.training_clients] == [1]
    assert [client.optimizer_steps for client in again.worker.training_clients] == [4]
    resumed = rows(run)[4]  # the second start's launch: a commit naming each policy's row of the first's final save
    assert resumed["name"] == "recovery-000002" and resumed["policies"]["worker"]["step"] == 8
    assert final["commit_seq"] == max(row["commit_seq"] for row in rows(run))
    tags = first.tags + second.tags
    for path in (run, run / "policies" / "worker"):
        assert _once(final_lineage(lineages, path)) and sorted(final_lineage(lineages, path)) == sorted(tags)


class MainSometimesAbsent(TaggedGroups):
    """The tagged groups, of which each whose tag 3 divides holds the worker's row only."""

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        output = await super().arun_episode(data)
        if output.samples[0].datum.model_input.to_ints()[0] % 3 == 0:
            return PolicyGroupOutput(output.samples[1:], OutputReport())
        return output


def test_groups_without_main_s_rows_count_toward_the_plan(tmp_path):
    """main's steps take every group, one without its rows too, so the run draws its plan, 4 groups a main step, and
    each row trains once into its policy's weights."""

    lineages = Lineages()
    _, workflow = start(tmp_path, lineages, steps=3, workflow=MainSometimesAbsent, timeout=120)
    run = run_dir(tmp_path)
    assert rows(run)[-1]["name"] == "final"
    without_main = {tag for tag in workflow.tags if tag % 3 == 0}
    assert len(workflow.tags) == 12 and len(without_main) == 4  # the plan: 3 steps of 4 groups, 4 without main's rows
    main, worker = final_lineage(lineages, run), final_lineage(lineages, run / "policies" / "worker")
    assert sorted(main) == sorted(set(workflow.tags) - without_main) and _once(main)
    assert sorted(worker) == sorted(workflow.tags) and _once(worker)


def test_an_interrupt_with_one_policy_s_step_open_lets_it_finish_and_saves_both(tmp_path):
    lineages = Lineages()
    # main's service asks for a stop in its second optimizer step: main's step is open, the worker's loop waits or
    # steps; each loop stops at its step's end, and the exit save names both.
    services, _ = start(tmp_path, lineages, steps=6, services=Services(lineages, stop_main=2))
    run = run_dir(tmp_path)
    interrupted = rows(run)[-1]
    assert interrupted["name"] == "000002_interrupted" and interrupted["batch"] == 2
    [main] = services.main.training_clients
    assert main.optimizer_steps == 2  # the open step finished
    worker_step = interrupted["policies"]["worker"]["step"]
    assert worker_step == services.worker.training_clients[0].optimizer_steps

    start(tmp_path, lineages, steps=6)
    # The groups that ran at the interrupt are drawn again; every share of the run's 24 groups trains once.
    for path in (run, run / "policies" / "worker"):
        trained = final_lineage(lineages, path)
        assert _once(trained) and len(trained) == 24


def test_a_stop_in_main_s_last_step_with_the_worker_behind_resumes_to_a_final_row_of_both(tmp_path):
    """The worker's optimizer steps are slow, and main's service asks for a stop in main's last optimizer step: the
    start saves both, the worker short of its last steps.  The resume starts with main at the plan's end: the worker
    trains its shares that are left, and the start saves the final row of both (it would otherwise save nothing, so
    that every later start trained them again)."""

    lineages = Lineages()
    services = Services(lineages, stop_main=3)
    services.worker.optim_delay = 0.05
    start(tmp_path, lineages, steps=3, services=services, timeout=120)
    run = run_dir(tmp_path)
    interrupted = rows(run)[-1]
    assert interrupted["name"] == "000003_interrupted" and interrupted["policies"]["worker"]["step"] < 12

    start(tmp_path, lineages, steps=3, timeout=120)
    final = rows(run)[-1]
    assert final["name"] == "final" and final["policies"]["worker"]["step"] == 12
    for path in (run, run / "policies" / "worker"):
        trained = final_lineage(lineages, path)
        assert _once(trained) and len(trained) == 12


# main steps on 8 shares in two updates of 4; the worker on 2 (an update each) within a bound of 1, so that its horizon
# of 4 shares holds dispatch back as soon as its loop stops taking them.
TWO_UPDATES: dict[str, Any] = {
    "train": {"batch_size": 8, "num_minibatches": 2},
    "policies": {"worker": {"batch_size": 2, "max_staleness": 1}},
}


@pytest.mark.parametrize("stop_at", [2, 10])
def test_an_interrupt_ends_when_an_open_step_needs_shares_that_the_stopped_policy_s_horizon_holds_back(
    tmp_path, stop_at
):
    """The worker's service asks for a stop in its optimizer step ``stop_at``, the last of one of its steps: its loop
    stops there, while main's step, which applied its first update, still needs shares.  Once the worker's horizon is
    full no group starts again, so main's step closes short with the shares it has, and the start saves both; the
    resume trains each share once, none dropped as stale."""

    lineages = Lineages()
    services = Services(lineages)
    services.worker.stop_at = stop_at
    start(tmp_path, lineages, steps=6, services=services, timeout=120, **TWO_UPDATES)
    run = run_dir(tmp_path)
    assert rows(run)[-1]["name"].endswith("_interrupted")
    assert rows(run / "policies" / "worker")[-1]["name"].endswith("_interrupted")

    logged: list[dict[str, Any]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(StatsLogger, "log_row", lambda self, stats, *, axis: logged.append(dict(stats)))
        start(tmp_path, lineages, steps=6, timeout=120, **TWO_UPDATES)
    for path in (run, run / "policies" / "worker"):
        assert _once(final_lineage(lineages, path))
    assert not any(value for stats in logged for key, value in stats.items() if key.endswith("stale_groups"))


class _Scored:
    """Cross entropy, which the fake service takes as any loss, with a metric of its own."""

    loss_fn = "cross_entropy"
    loss_fn_config: dict[str, Any] = {}
    side_channels: frozenset[str] = frozenset()

    def metrics(self, originals: list[Any], result: Any) -> dict[str, float]:
        return {"scored": 1.0}


class WorkerHook(px.StepHookBase):
    """A step hook that adds an update of the worker at the end of each of the worker's steps, and records whose step
    each end was (``ends``)."""

    name = "worker-hook"
    ends: ClassVar[list[str | None]] = []

    async def update(self, end: px.StepEnd) -> None:
        self.ends.append(end.policy)
        if end.policy == "worker":
            await end.train("worker", [checks._probe_datum()], objective=_Scored())


class CrossHook(px.StepHookBase):
    """A step hook that updates main at the end of the worker's steps, which independent stepping refuses."""

    name = "cross-hook"

    async def update(self, end: px.StepEnd) -> None:
        if end.policy == "worker":
            end.learner(MAIN, objective=_Scored())


register_component("step_hook", "test-independent-worker-hook", WorkerHook, exist_ok=True)
register_component("step_hook", "test-independent-cross-hook", CrossHook, exist_ok=True)


def test_a_step_hook_updates_the_policy_whose_step_ends_and_is_refused_another(tmp_path, monkeypatch):
    monkeypatch.setattr(WorkerHook, "ends", [])
    services, _ = start(
        tmp_path / "own", Lineages(), steps=1, trainer_options={"hooks": ["test-independent-worker-hook"]}
    )
    [worker] = services.worker.training_clients
    assert worker.optimizer_steps == 8  # its 4 steps' own updates and the hook's, at each end of a worker step
    assert Counter(WorkerHook.ends) == {MAIN: 1, "worker": 4}

    with pytest.raises(BaseException) as raised:
        start(tmp_path / "cross", Lineages(), steps=1, trainer_options={"hooks": ["test-independent-cross-hook"]})
    [error] = [error for error in _errors(raised.value) if isinstance(error, ValueError)]
    assert str(error).startswith("policy 'main' steps on its own clock: at the end of policy 'worker''s step a hook")


class RunLearnerHook(px.StepHookBase):
    """A step hook that, at the end of main's steps, sends an update of the worker through the run's learner of it
    rather than through its end, and records what happened (``seen``: the error's text, or ``sent``)."""

    name = "test-independent-run-learner-hook"
    seen: ClassVar[list[str]] = []
    # The run it was built for (from_run).
    run: Any = None

    @classmethod
    def from_run(cls, run: Any, options: Any) -> RunLearnerHook:
        hook = cls()
        hook.run = run
        return hook

    async def update(self, end: px.StepEnd) -> None:
        if end.policy != MAIN:
            return
        update = self.run.learner("worker").update()
        datum = checks._probe_datum()
        try:
            await update.submit([datum], [datum], minibatch=0, microbatch=0)
            await update.apply()
        except RuntimeError as error:
            self.seen.append(str(error))
            return
        self.seen.append("sent")


register_component("step_hook", RunLearnerHook.name, RunLearnerHook, exist_ok=True)


@pytest.mark.parametrize("trainer", ["rl", "ledger"])
def test_a_step_hook_reaches_no_other_policy_through_the_run_s_learner(tmp_path, monkeypatch, trainer):
    """At the end of main's step, the worker's own clock is not ending, but a hook still may not update the worker
    through ``run.learner``: the run's learners send nothing from a step end's hooks.  The worker's own steps, which a
    trainer of its own sends through ``run.learner`` meanwhile, train as before."""

    monkeypatch.setattr(RunLearnerHook, "seen", [])
    lineages = Lineages()
    if trainer == "ledger":
        start(tmp_path, lineages, steps=2, trainer=HookedLedgerTrainer, timeout=120)
    else:
        start(tmp_path, lineages, steps=2, trainer_options={"hooks": [RunLearnerHook.name]}, timeout=120)
    assert len(RunLearnerHook.seen) == 2, RunLearnerHook.seen  # one per main step
    assert all(seen.startswith("run.learner() is the trainer's, which sends nothing") for seen in RunLearnerHook.seen)
    worker = final_lineage(lineages, run_dir(tmp_path) / "policies" / "worker")
    assert _once(worker) and len(worker) == 8  # its every share, and no datum of the hook


def _errors(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [found for inner in error.exceptions for found in _errors(inner)]
    return [error]


class ExamplesHook(px.StepHookBase):
    """A step hook with state that trains examples of its own into the worker: at the end of each worker step, the next
    one of them, or two after an odd step, from its cursor.  Its state follows the worker's own steps, as a hook's
    state must when the policies step independently: it keeps the cursor after each worker step (``after``, by the
    step the worker resumes at from a row saved then), and ``resume`` takes the cursor at the step the worker resumes
    from (``start.policies["worker"].step``), not the newest, which a commit of main may have captured past the
    worker's row.  With ``keep``, the worker's recovery cadence, it drops the cursors no commit can name any more: it
    updates the worker at each of its steps, so the worker's newest row is never more than ``keep`` steps back."""

    name = "test-independent-examples-hook"
    keys = ("examples_hook_after",)
    # The examples' tags: BASE + the example's index.
    BASE = 900_000
    # How many of the worker's steps back it keeps cursors (None: every one).
    keep: ClassVar[int | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.after = {0: 0}

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {"examples_hook_after": {str(step): cursor for step, cursor in self.after.items()}}

    def resume(self, start: Any) -> None:
        saved = start.loop_state.get("examples_hook_after") or {"0": 0}
        step = start.policies["worker"].step  # where the worker resumes: its steps since then run again
        self.after = {int(at): cursor for at, cursor in saved.items() if int(at) <= step}

    async def update(self, end: px.StepEnd) -> None:
        if end.policy != "worker":
            return
        cursor, count = self.after[end.step], 1 + end.step % 2
        examples = [tagged_row(self.BASE + index, 0) for index in range(cursor, cursor + count)]
        await end.train("worker", examples, objective=_Scored())
        self.after[end.step + 1] = cursor + count
        if self.keep is not None:  # no commit names a row of the worker further back
            self.after = {at: cursor for at, cursor in self.after.items() if at >= end.step + 1 - self.keep}


register_component("step_hook", ExamplesHook.name, ExamplesHook, exist_ok=True)


class _Killed(BaseException):
    """The process died: nothing after it runs, no exit save."""


@pytest.mark.parametrize(("cadence", "keep"), [(100, None), (3, 3)])
def test_a_hook_whose_state_follows_each_policy_s_own_step_resumes_with_it(tmp_path, monkeypatch, cadence, keep):
    """The worker saves its launch row alone (or a row every 3 of its steps), and the process dies right after a commit
    of main whose capture took the hook's state after more worker steps than the worker's row it names: the resume
    repeats the worker's steps since that row, and the hook, resuming at the cursor of the worker's step, trains each
    of its examples into the worker once, in order (a hook that resumed at the cursor the commit saved would skip those
    of the repeated steps).  Keeping the cursors of its cadence only, every row holds at most that many and one."""

    monkeypatch.setattr(ExamplesHook, "keep", keep)
    lineages = Lineages()
    run = run_dir(tmp_path)
    sections: dict[str, Any] = {
        "policies": {"worker": {"checkpoint": {"recovery_every": cadence}}},
        "trainer_options": {"hooks": [ExamplesHook.name]},
    }
    original = commits_module.Commits._appended

    def appended(self: Any, of: str, own: Any, others: Any) -> None:
        original(self, of, own, others)
        if of == MAIN:
            saved = rows(run)[-1]["examples_hook_after"]  # the hook's state as main's commit captured it
            if max(int(at) for at in saved) > others["worker"]["step"]:
                raise _Killed()

    with monkeypatch.context() as armed:
        armed.setattr(commits_module.Commits, "_appended", appended)
        with pytest.raises(BaseException) as raised:
            start(tmp_path, lineages, steps=3, **sections)
        assert any(isinstance(error, _Killed) for error in _errors(raised.value)), raised.value
    start(tmp_path, lineages, steps=3, **sections)

    final = rows(run)[-1]
    steps = final["policies"]["worker"]["step"]
    trained = final_lineage(lineages, run / "policies" / "worker")
    examples = [tag - ExamplesHook.BASE for tag in trained if tag >= ExamplesHook.BASE]
    assert final["name"] == "final" and steps == 12
    assert examples == list(range(final["examples_hook_after"][str(steps)])) == list(range(18))
    assert _once([tag for tag in trained if tag < ExamplesHook.BASE])
    if keep is not None:  # every row of either policy holds the cursors of the worker's cadence only
        files = (run, run / "policies" / "worker")
        saved = [row.get("examples_hook_after", {}) for path in files for row in rows(path)]
        assert max(len(cursors) for cursors in saved) <= keep + 1, saved


class LedgerTrainer(px.TrainerBase):
    """A trainer of its own whose policies step independently: each policy's loop takes its shares from the share
    ledger, trains them through its learner, records them consumed, and ends its step on its own clock; a stop request
    ends each loop at its next take."""

    batches: ClassVar[dict[str, int]] = {MAIN: 4, "worker": 1}

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        return px.TrainerNeeds(rollouts=True, stepping="independent")

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        self.source = run.rollouts()
        return px.StepPlan(end_step=self.source.end_step)

    async def run_steps(self, run: Any) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                self.source.start(group, stop=run.shutdown)
                await asyncio.gather(*(self._loop(run, policy) for policy in self.batches))
                raise _LoopsEnded()  # the dispatcher's task ends with the group
        except* _LoopsEnded:
            pass

    async def _loop(self, run: Any, policy: str) -> None:
        ledger: px.ShareLedger = self.source.ledger
        clock = run.clocks[policy]
        while policy != MAIN or clock.step < run.plan.end_step:
            try:
                groups = await ledger.take(policy, self.batches[policy], stop=run.shutdown)
            except px.TrainInterrupted:
                run.interrupted = True
                return
            if not groups:
                return
            fates = self._fates(ledger, policy, groups)
            trained = [group for group in groups if group.key not in fates]
            if trained:
                update = run.learner(policy).update()
                for index, group in enumerate(trained):
                    datums = list(group.rollout)
                    await update.submit(datums, datums, minibatch=0, microbatch=index)
                await update.apply()
            async with ledger.lock:
                ledger.consumed(policy, [group.key for group in groups], step=clock.step, fates=fates)
            await run.end_step(policy=policy)

    def _fates(self, ledger: px.ShareLedger, policy: str, groups: list[Any]) -> dict[Any, str]:
        """The shares of ``groups`` the step drops, with their fates, by draw: none."""

        return {}


class _LoopsEnded(Exception):
    """Every policy's loop of :class:`LedgerTrainer` ended."""


class HookedLedgerTrainer(LedgerTrainer):
    """The ledger trainer, with :class:`RunLearnerHook` at its steps' ends."""

    def check(self, run: Any) -> None:
        run.hooks.register(RunLearnerHook.from_run(run, {}))


def test_a_trainer_of_its_own_steps_each_policy_through_the_share_ledger(tmp_path):
    lineages = Lineages()
    services, _ = start(tmp_path, lineages, steps=2, trainer=LedgerTrainer)
    assert [client.optimizer_steps for client in services.main.training_clients] == [2]
    run = run_dir(tmp_path)
    assert _once(final_lineage(lineages, run))
    worker_commits = [row for row in rows(run / "policies" / "worker") if "commit_seq" in row]
    assert worker_commits  # the worker committed on its own cadence, through the run's end_step
    assert services.worker.calls.count("create_lora_training_client fake-worker") == 1
    assert WORKER_URL in services.worker.sessions


def test_a_trainer_of_its_own_stops_its_loops_on_a_stop_request_and_resumes(tmp_path):
    lineages = Lineages()
    # main's service asks for a stop in its second optimizer step: each loop ends at its next take, the start saves.
    start(tmp_path, lineages, steps=4, trainer=LedgerTrainer, services=Services(lineages, stop_main=2), timeout=120)
    run = run_dir(tmp_path)
    assert rows(run)[-1]["name"] == "000002_interrupted"
    start(tmp_path, lineages, steps=4, trainer=LedgerTrainer, timeout=120)
    for path in (run, run / "policies" / "worker"):
        trained = final_lineage(lineages, path)
        assert _once(trained) and len(trained) == 16


class StaleLedgerTrainer(LedgerTrainer):
    """The ledger trainer, which drops each share whose rows are more versions of its policy old than the policy's
    bound (``bounds``), as the rl trainer's stale filter does, judged at the version the ledger gives
    (``judge_version``); ``dropped``: the tags of the shares it dropped, by policy."""

    bounds: ClassVar[dict[str, int]] = {MAIN: 1, "worker": 2}
    dropped: ClassVar[dict[str, list[int]]] = {}

    def _fates(self, ledger: px.ShareLedger, policy: str, groups: list[Any]) -> dict[Any, str]:
        fates = {}
        for group in groups:
            stamps = [int(datum.loss_fn_inputs["checkpoint_version"].to_torch()[0]) for datum in group.rollout]
            if ledger.judge_version(policy, group) - min(stamps) > self.bounds[policy]:
                fates[group.key] = "stale"
                self.dropped.setdefault(policy, []).append(group.rollout[0].model_input.to_ints()[0])
        return fates


def test_a_trainer_of_its_own_judges_a_share_a_resume_put_back_at_the_age_it_had(tmp_path, monkeypatch):
    """The worker saves its launch row alone, and the process dies right after a commit of main that names that row
    four or more published versions old: the resume puts back each share the worker consumed since, at one version
    more than the commit records it published.  Judged at the version the ledger gives each (its recorded age), none
    of them is dropped as stale, and each trains again; judged at the worker's version now, all of them would be."""

    monkeypatch.setattr(StaleLedgerTrainer, "dropped", {})
    lineages = Lineages()
    worker = {"max_staleness": 2, "checkpoint": {"recovery_every": 100}}
    original = commits_module.Commits._appended

    def appended(self: Any, of: str, own: Any, others: Any) -> None:
        original(self, of, own, others)
        if of == MAIN and others["worker"]["published_version"] - others["worker"]["policy_version"] >= 4:
            raise _Killed()

    with monkeypatch.context() as armed:
        armed.setattr(commits_module.Commits, "_appended", appended)
        with pytest.raises(BaseException) as raised:
            start(tmp_path, lineages, steps=4, trainer=StaleLedgerTrainer, policies={"worker": worker})
        assert any(isinstance(error, _Killed) for error in _errors(raised.value)), raised.value
    commit = resumed_from(tmp_path)
    named = commit.entries["worker"]
    assert commit.policy == MAIN and named["published_version"] - named["policy_version"] >= 4
    put_back = shares_put_back(tmp_path, commit)["worker"]
    assert len(put_back) >= 4

    monkeypatch.setattr(StaleLedgerTrainer, "dropped", {})
    start(tmp_path, lineages, steps=4, trainer=StaleLedgerTrainer, policies={"worker": worker})
    trained = final_lineage(lineages, run_dir(tmp_path) / "policies" / "worker")
    assert _once(trained) and set(put_back) <= set(trained)
    assert not set(put_back) & set(StaleLedgerTrainer.dropped.get("worker", []))


class TogetherTrainer(px.TrainerBase):
    """A trainer of its own whose policies step together (its needs say lockstep, whatever the config says)."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        return px.TrainerNeeds(rollouts=True)


def test_a_policy_s_own_batch_under_a_trainer_that_steps_its_policies_together_stops_the_start(tmp_path):
    with pytest.raises(ConfigError) as raised:
        start(tmp_path, Lineages(), steps=1, trainer=TogetherTrainer)
    assert raised.value.problems[0].startswith(
        "policies.worker.batch_size: a policy's own batch size is for policies that step on their own clocks"
    )
    assert "stepping together (in lockstep, as train.stepping or the trainer's needs() say)" in raised.value.problems[0]


# The config's stepping left at lockstep, without the worker's own batch and cadence: a trainer's needs choose.
LOCKSTEP_CONFIG: dict[str, Any] = {
    "train": {"stepping": "lockstep"},
    "policies": {"worker": {"batch_size": None, "checkpoint": None}},
}


class _Probed(Exception):
    """A trainer's start got as far as its steps."""


class ProbeTrainer(TogetherTrainer):
    """A trainer whose policies step together, which stops as its steps would begin."""

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        return px.StepPlan(end_step=1)

    async def run_steps(self, run: Any) -> None:
        raise _Probed()


def test_a_start_keeps_the_stepping_its_run_s_checkpoints_were_saved_with(tmp_path):
    """A trainer of its own whose needs say independent runs with train.stepping left at lockstep: its rows are
    commits.  A start whose trainer steps the policies together stops before its services open (exit 78), rather than
    train again what each policy consumed; and a start that steps independently stops on a run saved in lockstep."""

    lineages = Lineages()
    start(tmp_path / "independent", lineages, steps=1, trainer=LedgerTrainer, **LOCKSTEP_CONFIG)
    with pytest.raises(ConfigError, match="is a commit of trained policies that step independently"):
        start(tmp_path / "independent", lineages, steps=2, trainer=ProbeTrainer, **LOCKSTEP_CONFIG)

    with pytest.raises(BaseException) as raised:
        start(tmp_path / "lockstep", lineages, steps=1, trainer=ProbeTrainer, **LOCKSTEP_CONFIG)
    assert _errors(raised.value)[0].__class__ is _Probed  # it saved its launch, in lockstep
    with pytest.raises(ConfigError, match="no row holds a commit_seq"):
        start(tmp_path / "lockstep", lineages, steps=1, trainer=LedgerTrainer, **LOCKSTEP_CONFIG)


def test_a_trainer_whose_needs_step_independently_is_refused_a_group_admission_rule_whatever_the_config_says(tmp_path):
    lockstep: dict[str, Any] = {**LOCKSTEP_CONFIG, "train": {"stepping": "lockstep", "group_admission": "informative"}}
    with pytest.raises(ConfigError) as raised:
        start(tmp_path, Lineages(), steps=1, trainer=LedgerTrainer, **lockstep)
    assert raised.value.problems[0].startswith("train.stepping: independent steps each policy on its own clock")


class HalfwayHook(px.StepHookBase):
    """A step hook with state (a checkpoint participant): at the end of each worker step it adds 2 to ``n``, in two
    halves with an await between them, so that a checkpoint taken part-way saves an odd ``n``."""

    name = "test-independent-halfway-hook"
    keys = ("halfway_hook_n",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.n = 0

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {"halfway_hook_n": self.n}

    def resume(self, start: Any) -> None:
        self.n = int(start.loop_state.get("halfway_hook_n", 0))

    async def update(self, end: px.StepEnd) -> None:
        if end.policy == "worker":
            self.n += 1
            await asyncio.sleep(0.05)
            self.n += 1


register_component("step_hook", "test-independent-halfway-hook", HalfwayHook, exist_ok=True)


def test_no_commit_saves_a_hook_s_state_part_way_through_another_policy_s_step_end(tmp_path):
    """main commits at each of its steps while the worker's step ends run the hook: each commit waits until no step's
    end runs its hooks, so every n it saved is even (the worker's own commits save it after their step's hooks)."""

    start(tmp_path, Lineages(), steps=8, trainer_options={"hooks": ["test-independent-halfway-hook"]}, timeout=120)
    run = run_dir(tmp_path)
    saved = [row for path in (run, run / "policies" / "worker") for row in rows(path) if "halfway_hook_n" in row]
    assert len([row for row in saved if row["name"].startswith("recovery-0")]) > 4  # main's commits
    assert all(row["halfway_hook_n"] % 2 == 0 for row in saved), [(row["name"], row["halfway_hook_n"]) for row in saved]
