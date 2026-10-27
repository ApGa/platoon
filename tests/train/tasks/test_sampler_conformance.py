"""The conformance check of a task sampler, ``platoon.testing.check_task_sampler`` (RFC 0019, section 5): every
built-in passes it at each level it serves (``pass_rate`` with four sets of options, each given by name, so that its
decisions are replayed in processes with other hash seeds too), and so do the internal orders; and each of its named
checks catches what it checks, shown by a small broken sampler each.  The documented custom example passes it in
``tests/docs/test_extending_docs.py``, which takes it from the guide.

The built-ins are checked here without the timing check, which a machine busy with the parallel pass of the tests could
fail; ``pass_rate``'s decisions are timed against the check's budget in the serial pass, alone on the machine."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest

from platoon.experimental import Draw, Sampler, Slot
from platoon.testing import check_task_sampler
from platoon.testing.samplers import _cases, _Target, check_fast, run_checks
from platoon.train.samplers import PassRateOptions, PassRateSampler, _ListingOrder, _PositionOrder


@pytest.mark.parametrize(
    ("name", "levels", "options"),
    [
        ("weighted", ("run",), None),
        ("uniform", ("entry", "run"), None),
        ("pass_rate", ("entry", "run"), None),
        ("pass_rate", ("entry", "run"), {"max_draws_per_epoch": None}),
        ("pass_rate", ("entry", "run"), {"max_outstanding_per_task": 1}),
        ("pass_rate", ("entry", "run"), {"candidates": 16}),
    ],
)
def test_every_built_in_sampler_passes_the_check_at_each_level_it_serves(name, levels, options):
    check_task_sampler(name, options=options, levels=levels, budget_ms=None)


@pytest.mark.serial
@pytest.mark.parametrize("level", ["entry", "run"])
def test_pass_rate_decides_within_the_check_s_budget(level):
    """The check's ``fast`` alone, with its default budget (a median of 20 ms a decision on a pool of 4,096 tasks)."""

    target = _Target("pass_rate", PassRateSampler, ref="pass_rate")
    case = next(iter(_cases(target, PassRateOptions(), level, {})))
    assert check_fast(case, 20.0) == []


def test_the_internal_orders_pass_the_check():
    """The listing kept as it is, and an order handed in, which answer with core's internal row answer."""

    assert run_checks(_Target("listing", _ListingOrder), levels=("run",), internal=True) == {}
    rows: dict[int, dict[str, Any]] = {}

    def order(setup: Any, options: Any) -> _PositionOrder:
        def row(position: int) -> dict[str, Any]:
            return rows.setdefault(position, {"task_id": setup.task_ids[position * 7 % len(setup.task_ids)]})

        return _PositionOrder(setup, options, order=row)

    assert run_checks(_Target("order", _PositionOrder, factory=order), levels=("run",), internal=True) == {}


# --- each check catches what it checks ---------------------------------------------------------------------------


def failed(sampler: Any, **kwargs: Any) -> str:
    """The message of the check's AssertionError for ``sampler``."""

    with pytest.raises(AssertionError) as error:
        check_task_sampler(sampler, **kwargs)
    return str(error.value)


class Cycling(Sampler):
    """A correct static sampler: its pool's tasks in turn."""

    def next(self, ctx):
        return Draw(ctx.task_ids[ctx.decision % len(ctx.task_ids)], "cycle")


class NoLoad:
    adaptive = False

    def __init__(self, setup, options=None) -> None:
        self.setup = setup

    def next(self, ctx):
        return ctx.task_ids[0]

    def state_dict(self):
        return {}


def test_protocol_refuses_a_class_without_load_state_dict():
    assert "protocol: " in failed(NoLoad) and "load_state_dict" in failed(NoLoad)


class SlotsAndAdaptive(Sampler):
    adaptive = True
    entry_slots = True

    def next(self, ctx):
        return Slot("a")


def test_protocol_refuses_entry_slots_with_adaptive_and_at_the_entry_level():
    assert "hands slots to entries and is adaptive" in failed(SlotsAndAdaptive, levels=("run",))

    class Slots(Sampler):
        entry_slots = True

        def next(self, ctx):
            return Slot("a")

    message = failed(Slots, levels=("entry",))
    assert "protocol: " in message and "only a run's sampler does" in message


class Asynchronous(Sampler):
    async def next(self, ctx):
        return ctx.task_ids[0]


def test_protocol_refuses_a_coroutine_next():
    assert "coroutine" in failed(Asynchronous)


@dataclass(frozen=True)
class NeedsALimit:
    limit: int


class Unloadable(Cycling):
    options_schema = NeedsALimit


def test_options_refuses_a_schema_whose_defaults_do_not_load():
    message = failed(Unloadable, options={"limit": 3})
    assert "options: the default options: options.limit: required key is missing" in message


def test_options_refuses_a_typed_schema_that_takes_an_unknown_key():
    pydantic = pytest.importorskip("pydantic")

    class Lenient(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")
        x: int = 0

    class TakesAnything(Cycling):
        options_schema = Lenient

    message = failed(TakesAnything, levels=("entry",), budget_ms=None)
    assert "options: its options schema takes an unknown key" in message


class NoneWhileEligible(Sampler):
    adaptive = True

    def next(self, ctx):
        return None


class HeldOut(Sampler):
    def next(self, ctx):
        return "elsewhere::t0"


class EntrySlot(Sampler):
    def next(self, ctx):
        return Slot("a")


class NotANumber(Sampler):
    def next(self, ctx):
        return Draw(ctx.task_ids[0], "nan", math.nan)


class EmptyReason(Sampler):
    def next(self, ctx):
        return Draw(ctx.task_ids[0], "")


class Ineligible(Sampler):
    adaptive = True

    def next(self, ctx):
        return ctx.task_ids[0]


@pytest.mark.parametrize(
    ("sampler", "said"),
    [
        (NoneWhileEligible, "answered None at decision 0 while"),
        (HeldOut, "answered 'elsewhere::t0'"),
        (EntrySlot, "answered Slot(entry='a')"),
        (NotANumber, "score=nan"),
        (EmptyReason, "reason=''"),
        (Ineligible, "label x at its limit, which leaves entry a its w tasks: answered"),
    ],
)
def test_pool_task_or_nothing_refuses_what_the_dispatcher_refuses(sampler, said):
    message = failed(sampler, levels=("entry",), budget_ms=None)
    assert "pool-task-or-nothing: " in message and said in message


class IgnoresWhichAreEligible(Sampler):
    """Adaptive: None when nothing is eligible, else any task of its pool, whether or not its label has room."""

    adaptive = True

    def next(self, ctx):
        if not any(map(ctx.eligible, ctx.task_ids)):
            return None
        return ctx.task_ids[ctx.rng.randrange(len(ctx.task_ids))]


class KeepsItsFirstContext(Sampler):
    """Adaptive: it reads eligibility through the first decision's context, which a decision must not keep."""

    adaptive = True

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.eligible = None

    def next(self, ctx):
        self.eligible = self.eligible or ctx.eligible
        eligible = [task for task in ctx.task_ids if self.eligible(task)]
        return eligible[ctx.rng.randrange(len(eligible))] if eligible else None


def test_pool_task_or_nothing_refuses_an_ineligible_task_of_an_entry_and_an_earlier_decision_s_eligibility():
    """An entry whose tasks carry two labels, one of them at its limit, shows a sampler that ignores which of its
    tasks are eligible; one instance asked under the patterns in turn shows one that keeps a decision's context."""

    message = failed(IgnoresWhichAreEligible, levels=("entry",), budget_ms=None)
    assert "pool-task-or-nothing: " in message and "label x at its limit, which leaves entry a its w tasks" in message
    message = failed(KeepsItsFirstContext, levels=("entry",), budget_ms=None)
    assert "pool-task-or-nothing: " in message and "one instance under the patterns in turn: " in message


class Slow(Cycling):
    def next(self, ctx):
        if len(ctx.task_ids) > 1000:  # slow on the timed pool only, so that the other checks stay quick
            time.sleep(0.03)
        return super().next(ctx)


def test_fast_refuses_a_decision_slower_than_the_budget():
    message = failed(Slow, levels=("entry",), budget_ms=20.0)
    assert "fast: " in message and "more than the budget of 20 ms" in message


class SlowOnFourThousand(Cycling):
    """Slow only on a pool of more than 3,000 tasks: an entry's pool of 4,096 tasks, as the check times at the entry
    level too (its entries' half of a pool of 4,096 would pass)."""

    def next(self, ctx):
        if len(ctx.task_ids) > 3000:
            time.sleep(0.03)
        return super().next(ctx)


def test_fast_times_a_pool_of_4096_tasks_at_the_entry_level_too():
    message = failed(SlowOnFourThousand, levels=("entry",), budget_ms=20.0)
    assert "fast: " in message and "a decision on 4096 tasks takes" in message


class ReadsStatistics(Sampler):
    def next(self, ctx):
        ctx.stats.counts(ctx.task_ids[0])
        return ctx.task_ids[0]


class CountsCalls(Sampler):
    """Static, but its answer depends on how often it was asked, not on the decision index."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.calls = 0

    def next(self, ctx):
        self.calls += 1
        return ctx.task_ids[self.calls % len(ctx.task_ids)]


class StepDependent(Sampler):
    """Static, but its answer moves with the training steps committed."""

    def next(self, ctx):
        return ctx.task_ids[(ctx.decision + 7 * ctx.step) % len(ctx.task_ids)]


class VersionDependent(Sampler):
    """Static, but its answer moves with the policy version."""

    def next(self, ctx):
        return ctx.task_ids[(3 * ctx.decision + ctx.versions["main"]) % len(ctx.task_ids)]


def test_static_refuses_a_static_sampler_that_reads_statistics_or_depends_on_more_than_its_decision():
    assert "static: " in failed(ReadsStatistics, levels=("entry",), budget_ms=None)
    message = failed(CountsCalls, levels=("entry",), budget_ms=None)
    assert "static: " in message and "alone answered" in message
    for sampler in (StepDependent, VersionDependent):  # a resume recomputes a static sampler's counts at step 0
        message = failed(sampler, levels=("entry",), budget_ms=None)
        assert "static: " in message and "alone answered" in message


class SetState(Cycling):
    def state_dict(self):
        return {"seen": {"a", "b"}}

    def load_state_dict(self, state):
        pass


class KeyErrorState(Cycling):
    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.seen: list[str] = []

    def state_dict(self):
        return {"seen": list(self.seen)}

    def load_state_dict(self, state):
        self.seen = list(state["seen"])  # a malformed state raises KeyError, not ValueError


def test_state_refuses_a_state_that_is_not_json_and_a_load_that_raises_anything_but_value_error():
    assert "state: " in failed(SetState, levels=("entry",), budget_ms=None)
    message = failed(KeyErrorState, levels=("entry",), budget_ms=None)
    assert "state: " in message and "raised KeyError, not ValueError" in message


class ForgetsWhatItObserved(Sampler):
    """Adaptive: it draws the task of the group it observed last while that task is eligible, else its eligible tasks
    in turn, but saves only its decisions' count."""

    adaptive = True

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.last: str | None = None
        self.made = 0

    def observe(self, group):
        self.last = group.task_id

    def next(self, ctx):
        eligible = [task for task in ctx.task_ids if ctx.eligible(task)]
        if not eligible:
            return None
        self.made += 1
        return self.last if self.last in eligible else eligible[self.made % len(eligible)]

    def state_dict(self):
        return {"made": self.made}  # what it observed is left out

    def load_state_dict(self, state):
        if state and (set(state) != {"made"} or not isinstance(state["made"], int)):
            raise ValueError(f"not its state: {dict(state)!r}")
        self.made = state.get("made", 0) if state else 0


def test_state_refuses_a_state_that_leaves_out_what_the_sampler_observed():
    message = failed(ForgetsWhatItObserved, levels=("entry",), budget_ms=None)
    assert "state: " in message and "a fresh instance that loaded its state answered differently" in message


class RaisingObserver(Sampler):
    adaptive = True

    def next(self, ctx):
        return (
            next(task for task in ctx.task_ids if ctx.eligible(task)) if any(map(ctx.eligible, ctx.task_ids)) else None
        )

    def observe(self, group):
        raise RuntimeError("cannot observe")


class NeedsObserve(Sampler):
    adaptive = True

    def next(self, ctx):
        eligible = [task for task in ctx.task_ids if ctx.eligible(task)]
        if not eligible:
            return None
        return self.last  # set only by observe

    def observe(self, group):
        self.last = group.task_id


def test_observe_refuses_a_sampler_whose_observe_raises_or_that_needs_it():
    assert "observe: " in failed(RaisingObserver, levels=("entry",), budget_ms=None)
    message = failed(NeedsObserve, levels=("entry",), budget_ms=None)
    assert "observe: " in message and "without observe" in message


class NotFinite(Cycling):
    def metrics(self, stats):
        return {"rate": math.inf}


def test_metrics_refuses_a_value_that_is_not_a_finite_number():
    message = failed(NotFinite, levels=("entry",), budget_ms=None)
    assert "metrics: " in message and "'rate': inf" in message


class CountsFallbacksItself(Cycling):
    def metrics(self, stats):
        return {"draws/fallback": 0.0, "decision_ms": 0.0, "seen": 1.0}


def test_metrics_refuses_a_key_the_dispatcher_writes_itself():
    message = failed(CountsFallbacksItself, levels=("entry",), budget_ms=None)
    assert "metrics: " in message and "'draws/fallback', a key the dispatcher writes itself" in message
    assert "'decision_ms', a key the dispatcher writes itself" in message and "'seen'" not in message


SET_ORDER = """
from platoon.experimental import Sampler


class SetOrder(Sampler):
    def next(self, ctx):
        return sorted(ctx.task_ids)[ctx.decision % 3] if ctx.decision % 2 else next(iter(set(ctx.task_ids)))
"""


def test_deterministic_replays_a_sampler_given_by_reference_under_other_hash_seeds(tmp_path, monkeypatch):
    """Iterating a set of strings follows the process's hash seed: the replay in two other processes shows it."""

    (tmp_path / "set_order_sampler.py").write_text(SET_ORDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), *sys.path]))
    message = failed("set_order_sampler:SetOrder", levels=("entry",), budget_ms=None)
    assert "deterministic: " in message and "PYTHONHASHSEED" in message


def test_a_correct_custom_sampler_passes():
    check_task_sampler(Cycling, levels=("entry", "run"))


def test_check_task_sampler_is_exported_from_the_testing_kit():
    import platoon.testing

    assert "check_task_sampler" in platoon.testing.__all__
    assert platoon.testing.check_task_sampler is check_task_sampler
