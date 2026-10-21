"""The group admission API (RFC 0019, section 6): the answer's small print, registration and resolution, the built-in
rules' decisions (``informative``'s reasons in the order it checks, its band, a group without a verdict), and the
conformance check, ``platoon.testing.check_group_admission``: every built-in passes it, and so does a per-task quota
that reads the statistics; each named check catches what it checks, shown by a small broken rule each."""

from __future__ import annotations

import dataclasses
import os
import sys
import time

import pytest

from platoon.experimental import Admission, AdmissionRule, AdmissionSetup, GroupAdmission, group_admission
from platoon.extensions import ExtensionError
from platoon.registry import get_registry
from platoon.testing import check_group_admission
from platoon.train.admission_api import (
    REGISTRY_KIND,
    AdmissionError,
    checked_admission,
    groups_per_step,
    max_groups_problem,
    resolve_group_admission,
)
from platoon.train.admissions import AdmitAll, Informative, InformativeOptions
from platoon.train.sampler_api import TaskStatsView
from platoon.train.task_stats import MAIN, GroupRecord, MemberResult, TaskStatsStore

# --- the answer -------------------------------------------------------------------------------------------------------


def test_an_admission_s_reason_defaults_by_its_decision_and_a_bare_bool_reads_as_one():
    assert Admission(True).reason == "admitted"
    assert Admission(False).reason == "rejected"
    assert Admission(False, "uniform_pass").reason == "uniform_pass"
    assert checked_admission(False, "r") == Admission(False, "rejected")
    assert checked_admission(True, "r") == Admission(True, "admitted")


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (None, "not an Admission or a bool"),
        ("admit", "not an Admission or a bool"),
        (Admission(True, ""), "non-empty string of at most 64 characters"),
        (Admission(False, "x" * 65), "non-empty string of at most 64 characters"),
        (Admission(True, 3), "non-empty string"),  # type: ignore[arg-type]
    ],
)
def test_an_answer_outside_the_contract_is_an_admission_error(answer, said):
    with pytest.raises(AdmissionError, match=said):
        checked_admission(answer, "acme")


def test_the_cap_counts_whole_groups_rounded_as_a_person_reads_the_product():
    assert groups_per_step(3.0, 32) == 96
    assert groups_per_step(1.1, 10) == 11  # not 12: 1.1 × 10 is 11.000000000000002 in floats
    assert groups_per_step(1.0, 8) == 8
    assert groups_per_step(2.5, 3) == 8
    assert max_groups_problem(1) is None and max_groups_problem(3.0) is None
    for bad in (0.5, 0, -1, float("inf"), float("nan"), None, True, "3"):
        assert "must be a finite number of at least 1" in max_groups_problem(bad)


# --- registration and resolution --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["all", "informative"])
def test_core_s_names_are_reserved(name):
    with pytest.raises(ValueError, match="is core's"):
        group_admission(name)


@pytest.mark.parametrize(("name", "reads"), [("Informative", "informative"), ("ALL", "all"), ("All", "all")])
def test_a_name_that_reads_as_core_s_whatever_its_case_or_dashes_is_reserved_too(name, reads):
    """A config that misspells core's name (``Informative``) would otherwise pick up a package's rule of that name."""

    with pytest.raises(ValueError, match=f"reads as '{reads}', which is core's"):
        group_admission(name)


@pytest.mark.parametrize("name", ["a b", "mod:Rule", "rule@entry", ""])
def test_a_name_must_be_a_word_without_a_colon_or_an_at(name):
    with pytest.raises(ValueError, match="word without spaces"):
        group_admission(name)


def test_a_registered_name_resolves_with_its_options_and_check_and_a_second_registration_is_an_error():
    registry = get_registry(REGISTRY_KIND)
    name = "test-admission-api-quota"
    registry._items.pop(name, None)
    try:

        def refuse(options, ctx):
            raise ValueError("never")

        @group_admission(name, options=InformativeOptions, check=refuse)
        class Mine(AdmissionRule):
            def admit(self, group, stats):
                return True

        assert resolve_group_admission(name) is Mine
        assert Mine.options_schema is InformativeOptions
        with pytest.raises(ValueError, match="never"):
            Mine.check(None, None)
        with pytest.raises(ValueError, match="already has an entry"):
            group_admission(name)(Mine)
    finally:
        registry._items.pop(name, None)


def test_core_s_names_resolve_to_core_s_classes_whatever_is_registered():
    assert resolve_group_admission("all") is AdmitAll
    assert resolve_group_admission("informative") is Informative


def test_what_resolution_refuses():
    with pytest.raises(ExtensionError, match="unknown group admission rule 'informativ'; did you mean informative"):
        resolve_group_admission("informativ")
    with pytest.raises(ExtensionError, match="does not import"):
        resolve_group_admission("no_such_module_here:Rule")
    with pytest.raises(ExtensionError, match="is not a group admission rule: a class with admit"):
        resolve_group_admission("platoon.train.task_stats:GroupRecord")
    with pytest.raises(ExtensionError, match="coroutine"):
        resolve_group_admission(f"{__name__}:Asynchronous")


def test_the_built_ins_are_rules_of_the_protocol():
    setup = AdmissionSetup(name="all", level="run", entry=None, entries={}, batch_size=8, label_column="c")
    assert isinstance(AdmitAll(setup), GroupAdmission)
    assert isinstance(Informative(setup), GroupAdmission)
    assert Informative(setup).options == InformativeOptions()


# --- the built-ins' decisions -----------------------------------------------------------------------------------------


def member(*, scored: float | None, valid: bool = True) -> MemberResult:
    return MemberResult(
        reward=scored,
        scored=scored,
        valid=valid,
        outcome="scored" if valid else "error",
        disposition="completed",
        reward_key=None,
        steps=1,
        output_tokens=1,
        seconds=None,
        versions={MAIN: (0, 0)},
    )


def record(verdict: str, scores: list[float | None], *, entry: str = "e") -> GroupRecord:
    return GroupRecord(
        start=1,
        seq=0,
        task_id="e::t0",
        entry=entry,
        labels={},
        harness=None,
        group_size=len(scores),
        sampler="uniform",
        reason="order",
        score=None,
        position=0,
        drawn_at={MAIN: 0},
        members=tuple(member(scored=score, valid=score is not None) for score in scores),
        verdict=verdict,
    )


def stats(threshold: float = 1.0) -> TaskStatsView:
    from platoon.train.task_stats import EntryInfo

    store = TaskStatsStore(
        [("e::t0", "e", "e")], {"e": EntryInfo(group_size=4, success_threshold=threshold)}, column="c"
    )
    return TaskStatsView(store, now=0)


def informative(band=None) -> Informative:
    setup = AdmissionSetup(name="informative", level="run", entry=None, entries={}, batch_size=8, label_column="c")
    return Informative(setup, InformativeOptions(band=band))


@pytest.mark.parametrize(
    ("verdict", "scores", "band", "decision"),
    [
        ("uniform", [1.0, 1.0, 1.0, 1.0], None, Admission(False, "uniform_pass")),
        ("uniform", [0.0, 0.0, None, 0.0], None, Admission(False, "uniform_fail")),
        ("uniform", [1.0, 0.0, 1.0, 0.0], None, Admission(False, "uniform_mixed")),  # trained reward and success differ
        ("uniform", [None, None, None, None], None, Admission(False, "uniform_mixed")),
        ("reward_key_mismatch", [1.0, 0.0], None, Admission(False, "reward_key_mismatch")),
        ("no_results", [None, None], None, Admission(False, "no_results")),
        ("below_min", [1.0, None], None, Admission(False, "below_min")),
        ("no_completed_root", [None, None], None, Admission(False, "no_completed_root")),
        ("no_datums", [1.0, 0.0], None, Admission(False, "no_datums")),
        ("failed", [], None, Admission(False, "failed")),
        ("trained", [1.0, 0.0, 0.0, 0.0], None, Admission(True, "admitted")),
        ("trained", [1.0, 0.0, 0.0, 0.0], (0.3, 0.7), Admission(False, "below_band")),
        ("trained", [1.0, 1.0, 1.0, 0.0], (0.3, 0.7), Admission(False, "above_band")),
        ("trained", [1.0, 1.0, 0.0, 0.0], (0.3, 0.7), Admission(True, "admitted")),
        ("unknown", [], None, Admission(True, "unknown")),
        ("unknown", [], (0.3, 0.7), Admission(True, "unknown")),  # no valid member: no pass rate, the band is moot
        ("unknown", [1.0, 0.0, 0.0, 0.0], (0.3, 0.7), Admission(False, "below_band")),  # members give one
        ("a-verdict-platoon-does-not-know", [], None, Admission(True, "unknown")),
    ],
)
def test_informative_rejects_what_its_workflow_says_trains_nothing_then_the_band_and_admits_the_rest(
    verdict, scores, band, decision
):
    assert informative(band).admit(record(verdict, scores), stats()) == decision


def test_a_trained_group_that_all_passed_by_success_is_rejected_only_by_a_band():
    """Rewards 1.0 and 0.98 under a success threshold of 0.9: the advantages differ, the passes do not."""

    group = record("trained", [1.0, 0.98, 1.0, 0.98])
    assert informative().admit(group, stats(0.9)) == Admission(True)
    assert informative((0.2, 0.8)).admit(group, stats(0.9)) == Admission(False, "above_band")


@pytest.mark.parametrize("band", [[0.8, 0.2], [0.0, 1.5], [0.5], [0.1, 0.2, 0.3], [float("nan"), 0.5], [True, 0.5]])
def test_informative_s_band_is_two_numbers_lo_below_hi_between_zero_and_one(band):
    with pytest.raises(ValueError, match="band: must be two numbers lo < hi between 0 and 1"):
        InformativeOptions(band=band)


def test_all_admits_every_group():
    setup = AdmissionSetup(name="all", level="run", entry=None, entries={}, batch_size=8, label_column="c")
    for verdict in ("uniform", "trained", "no_datums", "unknown"):
        assert AdmitAll(setup).admit(record(verdict, [1.0, 1.0]), stats()) == Admission(True)


# --- the conformance check --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "options"), [("all", None), ("informative", None), ("informative", {"band": [0.2, 0.8]})]
)
def test_every_built_in_rule_passes_the_check(name, options):
    check_group_admission(name, options=options)


class Quota(AdmissionRule):
    """The extending guide's kind of rule: at most ``limit`` consumed groups of a task, read from its fates in the
    statistics (which count committed groups only: it can overshoot by the admitted groups not yet committed)."""

    @dataclasses.dataclass(frozen=True)
    class Options:
        limit: int = 2

    options_schema = Options

    def admit(self, group, stats):
        if (
            group.task_id in stats.task_ids()
            and stats.task(group.task_id).fates.get("consumed", 0) >= self.options.limit
        ):
            return Admission(False, "over_quota")
        return Admission(True)


def test_a_rule_that_reads_the_statistics_passes():
    check_group_admission(Quota, options={"limit": 1})


def failed(rule, **kwargs) -> str:
    with pytest.raises(AssertionError) as error:
        check_group_admission(rule, **kwargs)
    return str(error.value)


class CountsCalls(AdmissionRule):
    """Keeps state: it admits every other call."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.calls = 0

    def admit(self, group, stats):
        self.calls += 1
        return self.calls % 2 == 0


class ReturnsAString(AdmissionRule):
    def admit(self, group, stats):
        return "yes"


class Raises(AdmissionRule):
    def admit(self, group, stats):
        if group.verdict == "uniform":
            raise RuntimeError("cannot judge a uniform group")
        return True


class Asynchronous(AdmissionRule):
    async def admit(self, group, stats):  # type: ignore[override]
        return True


class NoAdmit:
    def __init__(self, setup, options=None) -> None:
        pass


class Slow(AdmissionRule):
    def admit(self, group, stats):
        if len(stats.task_ids()) > 1000:  # slow on the timed pool only, so that the other checks stay quick
            time.sleep(0.004)
        return True


class BadDefaults(AdmissionRule):
    @dataclasses.dataclass(frozen=True)
    class Options:
        needed: int

    options_schema = Options

    def admit(self, group, stats):
        return True


def test_pure_refuses_a_rule_that_keeps_state():
    message = failed(CountsCalls, budget_ms=None)
    assert "pure: " in message and ("asked twice" in message or "asked in reverse order" in message)


@pytest.mark.parametrize(
    ("rule", "said"),
    [
        (ReturnsAString, "answers: at the run level, group 0: group admission rule"),
        (Raises, "admit() raised on group"),
    ],
)
def test_answers_refuses_what_the_trainer_refuses(rule, said):
    message = failed(rule, budget_ms=None)
    assert "answers: " in message and said in message


# Rules that break the contract as a run would meet it, where a short check would not see it.


class CountedQuota(AdmissionRule):
    """Rejects a task's groups once this instance has admitted ``limit`` of them: a count kept in the rule, which a
    resume's new instance starts again from zero."""

    @dataclasses.dataclass(frozen=True)
    class Options:
        limit: int = 50

    options_schema = Options

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.admitted: dict[str, int] = {}

    def admit(self, group, stats):
        if self.admitted.get(group.task_id, 0) >= self.options.limit:
            return Admission(False, "over_quota")
        self.admitted[group.task_id] = self.admitted.get(group.task_id, 0) + 1
        return Admission(True)


class Budget(AdmissionRule):
    """Admits the first 1,000 groups it is asked about, then rejects every group."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.calls = 0

    def admit(self, group, stats):
        self.calls += 1
        return Admission(self.calls <= 1000, None if self.calls <= 1000 else "budget")


class FrozenView(AdmissionRule):
    """Rejects the groups of the tasks that had a consumed group when it was first asked: it keeps what it read of the
    statistics, so the newer views of later decisions never reach it."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.seen: frozenset[str] | None = None

    def admit(self, group, stats):
        if self.seen is None:
            self.seen = frozenset(task for task in stats.task_ids() if stats.task(task).fates.get("consumed", 0))
        return Admission(group.task_id not in self.seen, None if group.task_id not in self.seen else "seen_before")


class ReadsItsLabel(AdmissionRule):
    """Rejects label ``y``'s groups, reading the label as a mixture's groups carry it: a single environment's groups
    carry none."""

    def admit(self, group, stats):
        return Admission(group.labels[self.setup.label_column] != "y", "label_y")


class NeedsEntries(AdmissionRule):
    """Builds a table of the entries' group sizes, and refuses to be built without one: a trainer given an order
    without sampling has no entries."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.sizes = {label: info.group_size for label, info in setup.entries.items()}
        if not self.sizes:
            raise ValueError("no entries")

    def admit(self, group, stats):
        return True


class EntryLevelOnly(AdmissionRule):
    """Built at the entry level, it reads its entry's info by a misspelt attribute."""

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        if setup.level == "entry":
            self.size = setup.entries[setup.entry].group_sizes

    def admit(self, group, stats):
        return True


class BrokenCheck(AdmissionRule):
    """Its load-time check is an instance method: every config that names it would be refused with a TypeError."""

    def check(self, options, ctx):  # type: ignore[override]
        return None

    def admit(self, group, stats):
        return True


@pytest.mark.parametrize(
    ("rule", "options", "said"),
    [
        (CountedQuota, {"limit": 50}, "pure: at the run level, group"),
        (Budget, None, "asked of a new instance"),
        (FrozenView, None, "over a second view of the statistics"),
        (ReadsItsLabel, None, "answers: at the run level, admit() raised on group"),
        (NeedsEntries, None, "protocol: Class(setup, options) at the run level with no entries"),
        (EntryLevelOnly, None, "protocol: Class(setup, options) at the entry level raised AttributeError"),
        (BrokenCheck, None, "protocol: check(options, ctx) at the run level raised TypeError"),
    ],
)
def test_the_check_catches_state_and_failures_a_run_would_meet(rule, options, said):
    """A count or a budget kept in the rule shows after many calls, a view kept from an earlier call over another
    view, and a rule that fails on a single environment's groups, without entries, at the entry level, or in its
    load-time check, fails where a run would."""

    message = failed(rule, options=options, budget_ms=None)
    assert said in message, message


def test_a_rule_that_serves_one_level_is_checked_at_that_level_alone():
    check_group_admission(EntryLevelOnly, levels=("run",), budget_ms=None)
    assert "at the entry level" in failed(EntryLevelOnly, levels=("entry",), budget_ms=None)
    with pytest.raises(ValueError, match="levels: one or both of run, entry"):
        check_group_admission(AdmitAll, levels=("everywhere",))


def test_protocol_refuses_a_coroutine_and_a_class_without_admit_and_options_refuse_bad_defaults():
    assert "protocol: " in failed(Asynchronous) and "coroutine" in failed(Asynchronous)
    assert "protocol: " in failed(NoAdmit) and "a class with admit" in failed(NoAdmit)
    message = failed(BadDefaults, options={"needed": 1}, budget_ms=None)
    assert "options: the default options: options.needed: required key is missing" in message


def test_fast_refuses_a_decision_slower_than_the_budget():
    message = failed(Slow, budget_ms=2.0)
    assert "fast: " in message and "more than the budget of 2 ms" in message


SET_ORDER = """
from platoon.experimental import Admission, AdmissionRule


class SetOrder(AdmissionRule):
    def admit(self, group, stats):
        return Admission(True, next(iter({"first", "second", "third", "fourth"})))
"""


def test_pure_replays_a_rule_given_by_reference_under_other_hash_seeds(tmp_path, monkeypatch):
    """Iterating a set of strings follows the process's hash seed: the replay in two other processes shows it."""

    (tmp_path / "set_order_rule.py").write_text(SET_ORDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), *sys.path]))
    message = failed("set_order_rule:SetOrder", budget_ms=None)
    assert "pure: " in message and "PYTHONHASHSEED" in message


def test_check_group_admission_is_exported_from_the_testing_kit():
    import platoon.testing

    assert "check_group_admission" in platoon.testing.__all__
    assert platoon.testing.check_group_admission is check_group_admission
