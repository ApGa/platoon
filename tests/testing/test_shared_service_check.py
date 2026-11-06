"""``platoon.testing.check_shared_service`` on CPU, against a toy Tinker service whose LoRA adapters learn
(``shared_service_fakes.py``): a service that keeps two adapters apart passes; each fault that breaks one of RFC 0021's
U1 to U4 fails that check alone, by name, with what the check saw, and a check that raises an error stops the checks
after it; the check asks the service for what its arguments say, needs the Tinker SDK's telemetry, and ends every
session it opened; and the guide's example runs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from shared_service_fakes import FAULTS, GAIN, ToyModel, ToyService, toy_service

from platoon.testing import check_shared_service

URL = "http://tinker.invalid:8000"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
GUIDE = Path(__file__).resolve().parents[2] / "docs" / "how-to" / "test-extensions.md"
# The checks that do not run after U1 raises an error, as the report lists them.
AFTER_U1 = ("U2", "U3", "U4")
NOT_RUN_AFTER_U1 = "not run, after U1's error (the Tinker SDK ends the session of a call that fails)"


def failed(service: ToyService, *, not_run: Sequence[str] = (), **arguments) -> dict[str, list[str]]:
    """The checks that ``check_shared_service`` fails on ``service``, each with its lines; and that it lists the checks
    ``not_run`` as not run, after those, counts only the checks it failed, and ended and closed every session it
    opened (which a service that ignores session-end events leaves open)."""

    with toy_service(service), pytest.raises(AssertionError) as raised:
        check_shared_service(URL, base_model=MODEL, **arguments)
    header, *lines = str(raised.value).splitlines()
    checks: dict[str, list[str]] = {}
    for line in lines:
        name, what = line.strip().split(": ", 1)
        checks.setdefault(name, []).append(what)
    skipped = [name for name, found in checks.items() if found[0].startswith("not run")]
    failures = {name: found for name, found in checks.items() if name not in skipped}
    assert skipped == list(not_run) and list(checks) == [*failures, *skipped], checks
    assert header == f"the Tinker service at {URL} fails {len(failures)} check(s) of two adapters of {MODEL}:"
    assert all(session.closed for session in service.sessions)
    assert "ignores session end" in service.faults or all(session.ended for session in service.sessions)
    return failures


def test_a_service_that_keeps_two_adapters_apart_passes():
    service = ToyService()

    with toy_service(service):
        check_shared_service(URL, base_model=MODEL)

    # A's session, B's and A reopened's, all on the service's URL: A's ended by U3, and the others by the check's end,
    # which releases their models.
    assert [(session.base_url, session.ended, session.closed) for session in service.sessions] == [
        (URL, True, True)
    ] * 3
    assert service.created == [(MODEL, 32), (MODEL, 32)]
    assert {rate for _, rate in service.steps} == {0.0, 1e-4}
    assert [model.released for model in service.models] == [True, True, True]


def test_the_check_trains_at_the_rank_and_learning_rate_it_is_given():
    service = ToyService()

    with toy_service(service):
        check_shared_service(URL, base_model="other/base", rank=8, learning_rate=1e-3, tolerance=0.2)

    assert service.created == [("other/base", 8), ("other/base", 8)]
    assert {rate for _, rate in service.steps} == {0.0, 1e-3}


@pytest.mark.parametrize(
    ("fault", "check", "saw", "not_run"),
    [
        (
            "one adapter",
            "U1",
            "two LoRA adapters of Qwen/Qwen3-4B-Instruct-2507 at rank 32 did not open: RuntimeError",
            AFTER_U1,
        ),
        (
            "shared step",
            "U1",
            "a step of B at learning rate 0, beside A's at 0.0001, moved B by 0.200: A's step reached B",
            (),
        ),
        (
            "dropped gradients",
            "U1",
            "a step of A and B together at learning rate 0.0001 moved A's tokens by +0.200 and B's by +0.000, B's "
            "within the tolerance of 0.02: A's step dropped B's gradients",
            (),
        ),
        (
            "one gradient buffer",
            "U1",
            "a step of A and B together at learning rate 0.0001 failed: RuntimeError: model-1 holds gradients",
            AFTER_U1,
        ),
        ("sampler paths by name", "U2", "the sampler saves of A and B under one name are one path", ()),
        (
            "published base",
            "U2",
            "the samplers they publish set A and B apart by +0.000 on A's tokens and +0.000 on B's",
            (),
        ),
        ("end releases all", "U3", "reading B after A's session ended failed: RuntimeError: unknown model model-2", ()),
        (
            "ignores session end",
            "U3",
            "not checked: A's model still answers after its session's end event, so this service does not end a "
            "session on it",
            (),
        ),
        ("no optimizer on reopen", "U4", "a step with no gradient moved A reopened by +0.000, and A by +0.171", ()),
        ("no step count on reopen", "U4", "and A by +0.171: A did not reopen with its optimizer's state", ()),
        ("reopen into another model", "U4", "B's steps beside A's save and reopening moved B's tokens by -0.400", ()),
        (
            "NaN on reopen",
            "U4",
            "reading A reopened failed: ValueError: the mean logprob of A's tokens is nan, not a finite number",
            ("U3",),
        ),
    ],
)
def test_each_fault_fails_its_check_alone(fault: str, check: str, saw: str, not_run: tuple[str, ...]):
    assert f"({check})" in FAULTS[fault]

    failures = failed(ToyService(fault), not_run=not_run)

    assert list(failures) == [check]
    assert any(saw in line for line in failures[check]), failures[check]


def test_after_a_check_s_error_the_checks_after_it_do_not_run():
    # B's forward_backward fails in U1's first step, and B's session ends with it, as the Tinker SDK ends a session
    # after a call that fails: the checks after U1 would fail on B's released model.
    service = ToyService("one gradient buffer")

    with toy_service(service), pytest.raises(AssertionError) as raised:
        check_shared_service(URL, base_model=MODEL)

    assert str(raised.value).splitlines() == [
        f"the Tinker service at {URL} fails 1 check(s) of two adapters of {MODEL}:",
        "  U1: a step of A and B together at learning rate 0.0001 failed: RuntimeError: model-1 holds gradients: step "
        "it first",
        *(f"  {name}: {NOT_RUN_AFTER_U1}" for name in AFTER_U1),
    ]
    assert [model.released for model in service.models] == [True, True]


def test_a_failure_names_the_call_that_failed(monkeypatch):
    # B's model stops answering once A has saved its state, so U4's last reading of B fails, after A's steps.
    service = ToyService()
    forward = ToyModel.forward_async

    async def forward_unless_gone(model, *args, **kwargs):
        if model.model_id == "model-2" and service.states:
            raise RuntimeError("model-2 is gone")
        return await forward(model, *args, **kwargs)

    monkeypatch.setattr(ToyModel, "forward_async", forward_unless_gone)

    failures = failed(service, not_run=("U3",))

    assert failures == {
        "U4": [
            "reading B after its steps beside A's save and reopening failed: RuntimeError: model-2 is gone",
        ]
    }


def test_one_model_behind_both_adapters_fails_u1_and_more():
    failures = failed(ToyService("one model"))

    assert "A and B are one model, model-1" in failures["U1"]
    assert "U2" in failures and "U3" in failures


def test_every_check_that_fails_is_named_in_order():
    failures = failed(ToyService("end releases all", "shared step", "published base"))

    assert list(failures) == ["U1", "U2", "U3"]


def test_steps_too_small_to_see_fail_and_say_to_raise_the_learning_rate():
    failures = failed(ToyService(), learning_rate=0.5e-5)  # each step moves a token's logprob by 0.01

    moved = f"{0.5e-5 * GAIN:+.3f}"
    assert failures["U1"] == [
        f"a step of A and B together at learning rate 5e-06 moved A's tokens by {moved} and B's by {moved}, within "
        "the tolerance of 0.02 (raise learning_rate, if the service trains at all)"
    ]
    assert "too little to check their samplers (raise learning_rate)" in failures["U2"][0]
    assert "raise learning_rate" in failures["U4"][0]


def test_without_a_url_it_names_the_service_tinker_base_url_names(monkeypatch):
    monkeypatch.setenv("TINKER_BASE_URL", "http://from-the-environment:8000")
    service = ToyService("published base")

    with toy_service(service), pytest.raises(AssertionError, match="^the Tinker service at http://from-the-env"):
        check_shared_service(None, base_model=MODEL)

    assert [session.base_url for session in service.sessions] == [None] * 3


def test_without_the_sdk_s_telemetry_it_refuses_before_any_adapter_opens():
    # With TINKER_TELEMETRY=0 the SDK sends no session-end event: the check could end no session, and a failed call
    # would leave its session open.
    service = ToyService(telemetry=False)

    with toy_service(service), pytest.raises(RuntimeError, match="needs the Tinker SDK's telemetry, which TINKER_TELE"):
        check_shared_service(URL, base_model=MODEL)

    assert service.created == []
    assert [(session.ended, session.closed) for session in service.sessions] == [(False, True)]


@pytest.mark.parametrize("arguments", [{"learning_rate": 0.0}, {"tolerance": -1.0}])
def test_a_learning_rate_or_tolerance_that_is_not_positive_is_refused(arguments):
    service = ToyService()

    with toy_service(service), pytest.raises(ValueError, match="must be positive"):
        check_shared_service(URL, base_model=MODEL, **arguments)

    assert service.sessions == []


def test_the_guide_s_example_runs():
    section = GUIDE.read_text().split("\n## Check a shared Tinker service\n", 1)[1].split("\n## ", 1)[0]
    (block,) = re.findall(r"^```python\n(.*?)^```$", section, re.S | re.M)

    with toy_service(ToyService()):
        exec(compile(block, "test-extensions.md", "exec"), {})
