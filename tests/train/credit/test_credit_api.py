"""The credit API (RFC 0021, section 6): a row's advantages set on its action tokens only, the ``credit`` component
kind (core's names, reserved names, registered names, ``module:Class``, and what is no estimator), and what the run
calls of the estimators: their settle as a group is filed, their state as a checkpoint takes it and a resume gives it
back, and the error that stops the run when one of them raises."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import tinker
import torch
from tinker import TensorData

from platoon import registry as registry_module
from platoon.extensions import ExtensionError
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import CreditGroup, TrainingSample
from platoon.train.credit.group import GroupCredit
from platoon.train.credit.keyed import Keyed
from platoon.train.credit.registry import REGISTRY_KIND, credit_estimator, estimator_problem, resolve_credit_estimator
from platoon.train.credit.runtime import (
    CREDIT_KEY,
    Credit,
    CreditEstimatorError,
    credit_states,
    load_credit_states,
    settle_credit,
)
from platoon.train.policy.spec import MAIN
from platoon.utils.stats_tracker import DeferredStats


def _sample(mask: list[float], advantages: list[float]) -> TrainingSample:
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(mask)))),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor(mask)),
            "advantages": TensorData.from_torch(torch.tensor(advantages)),
        },
    )
    return TrainingSample(datum, "main", 0)


@pytest.fixture
def registry(monkeypatch):
    """A fresh registry of credit estimators, so that the names a test registers go with it."""

    monkeypatch.setattr(registry_module, "_REGISTRIES", {**registry_module._REGISTRIES})
    registry_module._REGISTRIES.pop(REGISTRY_KIND, None)


# --- rows ------------------------------------------------------------------------------------------------------------


def test_set_advantage_sets_the_action_tokens_and_leaves_the_others():
    sample = _sample([0.0, 1.0, 1.0, 0.0], [0.5, 1.0, 1.0, 0.25])
    sample.set_advantage(-0.5)
    assert sample.advantages.tolist() == [0.5, -0.5, -0.5, 0.25]
    assert sample.advantages.dtype == torch.float32


def test_set_advantages_takes_one_value_per_target_token_and_sets_the_action_tokens_only():
    sample = _sample([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0])
    sample.set_advantages([9.0, 0.25, 0.75, 9.0])
    assert sample.advantages.tolist() == [0.0, 0.25, 0.75, 0.0]
    with pytest.raises(ValueError, match=r"\(3,\) values for a row of \(4,\) target tokens"):
        sample.set_advantages([1.0, 2.0, 3.0])


# --- the component kind ----------------------------------------------------------------------------------------------


def test_core_names_resolve_to_core_s_estimators():
    assert resolve_credit_estimator("group") is GroupCredit
    assert resolve_credit_estimator("keyed") is Keyed
    assert GroupCredit.needs_agents is False and Keyed.needs_agents is True


@pytest.mark.parametrize("name", ["reverse_kl", "gae", "Reverse-KL"])
def test_the_names_platoon_adds_are_reserved(name, registry):
    with pytest.raises(ExtensionError, match="reserved for one platoon will add"):
        resolve_credit_estimator(name)
    with pytest.raises(ValueError, match="is core's"):
        credit_estimator(name)


def test_an_unknown_name_says_what_resolves_and_the_nearest(registry):
    with pytest.raises(ExtensionError, match=r"unknown credit estimator 'keyd'; did you mean keyed\?"):
        resolve_credit_estimator("keyd")
    with pytest.raises(ExtensionError, match=r"unknown credit estimator 'Keyed'; did you mean keyed\?"):
        resolve_credit_estimator("Keyed")


class _ByEntry:
    needs_agents = False

    def __init__(self, setup: Any, options: Any = None) -> None:
        self.setup = setup

    async def assign(self, group: Any) -> None:
        pass


def test_a_registered_name_and_module_class_resolve(registry):
    credit_estimator("acme-by-entry")(_ByEntry)
    assert resolve_credit_estimator("acme-by-entry") is _ByEntry
    assert resolve_credit_estimator(f"{__name__}:_ByEntry") is _ByEntry
    with pytest.raises(ValueError, match="already has an entry named 'acme-by-entry'"):
        credit_estimator("acme-by-entry")(_ByEntry)
    with pytest.raises(ValueError, match="a word without spaces"):
        credit_estimator("acme:by")


class _Sync:
    needs_agents = False

    def assign(self, group: Any) -> None: ...


class _NoFlag:
    async def assign(self, group: Any) -> None: ...


class _AsyncSettle(_ByEntry):
    async def settle(self, group: Any) -> None: ...


class _HalfState(_ByEntry):
    def state(self) -> dict:
        return {}


@pytest.mark.parametrize(
    ("cls", "problem"),
    [
        (_Sync, "a class with async assign(group)"),
        (_NoFlag, "its needs_agents is no bool"),
        (_AsyncSettle, "its settle(group) is no synchronous method"),
        (_HalfState, "one of state() and load_state(state) without the other"),
    ],
)
def test_what_is_no_estimator_is_refused_with_why(cls, problem):
    assert problem in (estimator_problem(cls) or "")
    with pytest.raises(ExtensionError, match="is not a credit estimator"):
        resolve_credit_estimator(f"{__name__}:{cls.__name__}")


# --- what the run calls ----------------------------------------------------------------------------------------------


class _Output(list):
    credit: Credit | None = None


class _Counting(_ByEntry):
    def __init__(self, setup: Any = None, options: Any = None) -> None:
        self.setup = setup
        self.seen: list[str] = []

    def settle(self, group: CreditGroup) -> None:
        self.seen.append(group.task_id)

    def state(self) -> dict:
        return {"seen": self.seen}

    def load_state(self, state: dict) -> None:
        self.seen = list(state["seen"])


def _credit(estimator: Any, task_id: str = "g1") -> Credit:
    group = CreditGroup(task_id, None, "a", WorkflowConfig(), [], {}, {MAIN: 0}, DeferredStats())
    return Credit(estimator, group, ())


def test_settle_credit_settles_the_output_s_credit_once_and_nothing_for_other_outputs():
    counting = _Counting()
    output = _Output()
    output.credit = _credit(counting)
    settle_credit(output)
    settle_credit(output)
    assert counting.seen == ["g1"] and output.credit is None
    settle_credit([])  # a plain list of datums
    settle_credit(None)  # a failed group
    settle_credit(_Output())


def test_credit_states_are_json_copies_by_entry_with_their_class_and_load_back():
    counting, plain = _Counting(), _ByEntry(None)
    workflow = SimpleNamespace(entry_credit={"a": counting, "b": plain})
    counting.seen.append("g1")
    states = credit_states(workflow)
    counting.seen.append("g2")  # after the capture: the capture keeps what it took
    assert states == {"a": {"class": f"{__name__}:_Counting", "state": {"seen": ["g1"]}}}
    resumed = _Counting()
    load_credit_states(SimpleNamespace(entry_credit={"a": resumed}), states)
    assert resumed.seen == ["g1"]
    assert credit_states(SimpleNamespace(entry_credit={"b": plain})) is None
    assert credit_states(SimpleNamespace()) is None and CREDIT_KEY == "credit"


class _Mean(_ByEntry):
    """Another estimator with state, of another shape than _Counting's."""

    mean = 0.0

    def state(self) -> dict:
        return {"mean": self.mean}

    def load_state(self, state: dict) -> None:
        self.mean = float(state["mean"])


def test_a_resume_gives_a_state_only_to_an_estimator_of_the_class_that_saved_it(caplog):
    counting = _Counting()
    counting.seen.append("g1")
    states = credit_states(SimpleNamespace(entry_credit={"a": counting}))
    other = _Mean(None)
    load_credit_states(SimpleNamespace(entry_credit={"a": other}), states)  # the entry's credit changed
    assert other.mean == 0.0
    warning = "environments[a] starts afresh: the checkpoint holds the state of another credit estimator"
    assert f"{warning}, {__name__}:_Counting" in caplog.text


class _Unserializable(_Counting):
    def state(self) -> dict:
        return {"seen": {1, 2}}


class _Unloadable(_Counting):
    def load_state(self, state: dict) -> None:
        raise KeyError("mean")


def test_an_estimator_that_raises_stops_the_run_with_an_error_that_names_it_and_its_entry():
    with pytest.raises(CreditEstimatorError, match=r"_Unserializable of environments\[hard\]: state\(\) gave no JSON"):
        credit_states(SimpleNamespace(entry_credit={"hard": _Unserializable()}))
    saved = {"a": {"class": f"{__name__}:_Unloadable", "state": {}}}
    with pytest.raises(CreditEstimatorError, match=r"environments\[a\]: load_state\(state\) raised KeyError: 'mean'"):
        load_credit_states(SimpleNamespace(entry_credit={"a": _Unloadable()}), saved)

    class Broken(_Counting):
        def settle(self, group: CreditGroup) -> None:
            raise RuntimeError("a bug in settle")

    output = _Output()
    output.credit = _credit(Broken(SimpleNamespace(name="acme-broken")), "task-7")
    with pytest.raises(
        CreditEstimatorError,
        match=r"^the credit estimator acme-broken of environments\[a\]: settle\(group\) raised RuntimeError: a bug "
        r"in settle, on task task-7$",
    ):
        settle_credit(output)


def test_only_a_class_registers(registry):
    def assign(group: Any) -> None: ...

    with pytest.raises(TypeError, match=r"@credit_estimator\('acme-fn'\) registers a class"):
        credit_estimator("acme-fn")(cast(Any, assign))
