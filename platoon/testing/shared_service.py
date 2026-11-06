"""The check of a Tinker service on which a run's LoRA adapters share one base model (RFC 0021, section 12):
:func:`check_shared_service`.

Trained policies with one ``model_name`` on one service are two LoRA adapters there, each in a Tinker session of its
own.  :func:`check_shared_service` opens two such adapters, A and B, on a live service, and checks what a run of them
needs of it, RFC 0021's U1 to U4 (:data:`CHECKS`); its ``AssertionError`` lists each check that failed, by name, with
what it saw:

- ``U1``: the two open at once, as two models, and each steps on its own: a step of the two together at the check's
  learning rate (every ``forward_backward``, then every ``optim_step``, submitted before any result is awaited, as a
  step of several policies submits them) moves each, which fails on a service where a step of one drops the other's
  gradients; and a step of one at the check's learning rate, beside a step of the other at learning rate 0, moves the
  one and leaves the other as it was, each way round;
- ``U2``: each adapter's sampler serves its own weights: the samplers of their sampler saves under one name (as a
  checkpoint of several policies saves them), and the samplers they publish (``save_weights_and_get_sampling_client``,
  as each step publishes), set A and B apart as the adapters' weights do;
- ``U3``: one session's end leaves the other sessions' models: once A's session ends, as the Tinker SDK ends one after a
  fatal error (its session-end event, then no heartbeat), and A's model no longer answers, B and the A that U4 reopened
  read as before and still step, B's sampler serves as before, and B still saves.  A service on which A's model still
  answers after that event does not end a session on it, and fails U3 as not checked;
- ``U4``: A, saved with its optimizer while B steps, reopens in a session of its own (as a resume opens it) while B
  steps again, with its weights and its optimizer's state: it reads as A did at the save, and a step with no gradient
  (every loss weight 0) moves it by its momentum about as far as the same step moved A; and B's steps are kept.

Each adapter trains on 16 token ids of its own (:data:`TOKENS`, ``cross_entropy``), and the check reads an adapter's
weights as the mean logprob they give each adapter's tokens, through its training client (``forward``) or through a
sampler (``compute_logprobs``); a reading that is not a finite number is an error.  It compares readings of one kind
only, and judges the differences that U2 and U4 compare only once they are :data:`MARGIN` tolerances wide.

A check that raises an error fails with what the check was doing when it did, and the checks after it do not run (the
report lists them as not run, after the checks that failed): the Tinker SDK ends the session of a call that fails,
and a service may release the session's models then.  The check runs U3 last, since that ends a session, and at the
end it ends every session it opened as U3 ends A's.  It needs the SDK's telemetry, which sends the session-end event,
and raises ``RuntimeError`` before any adapter opens when ``TINKER_TELEMETRY`` turns it off.

It checks what the experimental support for several policies (RFC 0021) asks of a service, and changes with it.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Sequence
from typing import Any

# The checks, by name, with what each asks of the service (RFC 0021, section 12), in the order they are reported.
CHECKS = {
    "U1": "several LoRA models of one base, each with its own Adam state, stepping separately",
    "U2": "each adapter's sampler serves its own weights",
    "U3": "one session's end leaves the other sessions' models",
    "U4": "save one adapter and reopen it with its optimizer while another trains",
}
# The token ids each adapter trains on: ordinary ids of any model's vocabulary, and none of them in both sequences.
TOKENS = {"A": tuple(range(1000, 1016)), "B": tuple(range(2000, 2016))}
# The name of the check's saves, which it leaves on the service.
SAVE_NAME = "platoon-check-shared-service"
# How many tolerances wide a difference must be before U2 or U4 judges it: each asks for half of it, which then still
# stands out from the noise of the four readings that a comparison takes.
MARGIN = 4

# One optimizer step of a training client: (client, whose tokens it trains on, their loss weight, learning rate).
_Step = tuple[Any, str, float, float]
# The error that the check's session-end events report, as the Tinker SDK reports the error that ends a session.
_ENDING = "platoon.testing.check_shared_service ends this session"
_NO_TELEMETRY = (
    "check_shared_service needs the Tinker SDK's telemetry, which TINKER_TELEMETRY turns off (unset it, or set it "
    "to 1): the session-end event it sends is how the check ends a session, and how the SDK ends one after a call "
    "that fails"
)


def check_shared_service(
    url: str | None,
    *,
    base_model: str,
    rank: int = 32,
    learning_rate: float = 1e-4,
    tolerance: float = 0.02,
) -> None:
    """Check that the Tinker service at ``url`` (None: ``TINKER_BASE_URL``, else hosted Tinker) serves two LoRA
    adapters of ``base_model`` at ``rank`` as RFC 0021's U1 to U4 ask (:data:`CHECKS`); raise ``AssertionError`` naming
    every check it fails, with what it saw, and the checks it did not run after one that raised an error.  Its
    optimizer steps are Adam's at ``learning_rate``; ``tolerance`` is how far two readings of the same weights may
    differ, in mean logprob per token, and each of its steps must move its adapter further than that.  It holds three
    adapters at once (A, B and A reopened), each in a session of its own, ends every session it opened, and leaves its
    saves on the service (named ``platoon-check-shared-service...``).  It needs the Tinker SDK's telemetry, and raises
    ``RuntimeError`` before any adapter opens when ``TINKER_TELEMETRY`` turns it off.  Call it outside a running event
    loop."""

    if not (learning_rate > 0 and tolerance > 0):
        raise ValueError(f"learning_rate and tolerance must be positive, not {learning_rate} and {tolerance}")
    failures, not_run = asyncio.run(_Check(url, base_model, rank, learning_rate, tolerance).run())
    if failures:
        where = url if url is not None else os.environ.get("TINKER_BASE_URL") or "hosted Tinker"
        lines = [f"{name}: {what}" for name in CHECKS for what in failures.get(name, ())] + [
            f"{name}: not run, after {not_run[name]}'s error (the Tinker SDK ends the session of a call that fails)"
            for name in CHECKS
            if name in not_run
        ]
        raise AssertionError(
            f"the Tinker service at {where} fails {len(failures)} check(s) of two adapters of {base_model}:\n  "
            + "\n  ".join(lines)
        )


class _NoTelemetry(Exception):
    """A session of the check has no telemetry (``TINKER_TELEMETRY``)."""


class _Check:
    """One run of the checks: its adapters (``clients``: A and B) and their sessions, every session it opened and
    those it closed, A reopened (by U4), and what it is doing, which a failure names."""

    def __init__(self, url: str | None, base_model: str, rank: int, learning_rate: float, tolerance: float) -> None:
        self.url, self.base_model, self.rank = url, base_model, rank
        self.learning_rate, self.tolerance = learning_rate, tolerance
        self.clients: dict[str, Any] = {}
        self.session_of: dict[str, Any] = {}
        self.sessions: list[Any] = []
        self.closed: list[Any] = []
        self.reopened: Any = None
        self.doing = ""

    async def run(self) -> tuple[dict[str, list[str]], dict[str, str]]:
        """The problems of each check that failed, by name, and each check that did not run, with the check whose error
        stopped it; no check runs when the service does not open both adapters."""

        try:
            if found := await self.open():
                return {"U1": found}, dict.fromkeys(("U2", "U4", "U3"), "U1")
            failures: dict[str, list[str]] = {}
            not_run: dict[str, str] = {}
            error = None
            for name, check in (("U1", self.u1), ("U2", self.u2), ("U4", self.u4), ("U3", self.u3)):
                if error is not None:
                    not_run[name] = error
                    continue
                try:
                    found = await check()
                except Exception as exc:  # noqa: BLE001 - a check reports what failed
                    found, error = [f"{self.doing} failed: {_error(exc)}"], name
                if found:
                    failures[name] = found
            return failures, not_run
        finally:
            await self.end_sessions()

    async def open(self) -> list[str]:
        """Open A and B, each in a session of its own, as a run opens each policy's: U1's problems when they do not
        open.  Raises ``RuntimeError``, before any adapter opens, when the SDK's telemetry is off."""

        try:
            for name in TOKENS:
                session = self.session_of[name] = self.session()
                if session.get_telemetry() is None:
                    raise _NoTelemetry
                self.clients[name] = await session.create_lora_training_client_async(self.base_model, rank=self.rank)
        except _NoTelemetry:
            raise RuntimeError(_NO_TELEMETRY) from None
        except Exception as exc:  # noqa: BLE001 - a check reports what failed
            return [f"two LoRA adapters of {self.base_model} at rank {self.rank} did not open: {_error(exc)}"]
        return []

    def session(self) -> Any:
        """A new session on the service, as each policy of a run opens one of its own."""

        import tinker

        session = tinker.ServiceClient(base_url=self.url)
        self.sessions.append(session)
        return session

    def close(self, session: Any) -> None:
        """Stop ``session``'s heartbeat and its telemetry."""

        session.holder.close()
        self.closed.append(session)

    async def end_sessions(self) -> None:
        """End every session the check opened that is still open, as U3 ends A's: its session-end event, then no
        heartbeat (a service that does not end a session on that event keeps its models until its lease expires)."""

        async def end(session: Any) -> None:
            try:
                await _end_event(session)
            except Exception:  # noqa: BLE001 - the session closes all the same
                pass
            finally:
                self.close(session)

        await asyncio.gather(*(end(session) for session in self.sessions if session not in self.closed))

    async def reading(self, name: str, client: Any, when: str = "", *, sampler: bool = False) -> dict[str, float]:
        """The reading of ``client``, a training client or a sampler, taken while the check is "reading <name>
        <when>"."""

        self.doing = f"reading {name} {when}".rstrip()
        return await (_served(client) if sampler else _read(client))

    async def readings(self, when: str) -> dict[str, dict[str, float]]:
        """A's and B's readings, through their training clients, each taken while the check is "reading <name>
        <when>"."""

        return {name: await self.reading(name, client, when) for name, client in self.clients.items()}

    async def u1(self) -> list[str]:
        a, b = self.clients["A"], self.clients["B"]
        lr, tolerance = self.learning_rate, self.tolerance
        problems = []
        if getattr(a, "model_id", None) is not None and a.model_id == getattr(b, "model_id", None):
            problems.append(f"A and B are one model, {a.model_id}")
        # A step of the two fresh adapters, whose optimizers hold no momentum yet: one whose gradients the other's step
        # drops moves by nothing.
        readings = await self.readings("before its first step")
        self.doing = f"a step of A and B together at learning rate {lr:g}"
        await _step([(a, "A", 1.0, lr), (b, "B", 1.0, lr)])
        after = await self.readings("after the step of A and B together")
        rose = {name: after[name][name] - readings[name][name] for name in TOKENS}
        together = (
            f"a step of A and B together at learning rate {lr:g} moved A's tokens by {rose['A']:+.3f} and B's by "
            f"{rose['B']:+.3f}"
        )
        if not any(rose[name] > tolerance for name in TOKENS):
            too_small = f"within the tolerance of {tolerance:g} (raise learning_rate, if the service trains at all)"
            return [*problems, f"{together}, {too_small}"]
        for name, other in (("A", "B"), ("B", "A")):
            if not rose[name] > tolerance:
                problems.append(
                    f"{together}, {name}'s within the tolerance of {tolerance:g}: {other}'s step dropped {name}'s "
                    f"gradients, or {name}'s step was lost beside it"
                )
        readings = after
        for mover, still in (("A", "B"), ("B", "A")):
            beside = f"a step of {mover} beside one of {still} at learning rate 0"
            self.doing = beside
            await _step([(self.clients[mover], mover, 1.0, lr), (self.clients[still], still, 1.0, 0.0)])
            after = await self.readings(f"after {beside}")
            if not (rose_mover := after[mover][mover] - readings[mover][mover]) > tolerance:
                problems.append(
                    f"a step of {mover} at learning rate {lr:g}, beside one of {still} at learning rate 0, moved "
                    f"{mover}'s tokens by {rose_mover:+.3f}, within the tolerance of {tolerance:g}: {mover}'s step "
                    f"took {still}'s learning rate, or was lost beside it"
                )
            if (moved := _distance(after[still], readings[still])) > tolerance:
                problems.append(
                    f"a step of {still} at learning rate 0, beside {mover}'s at {lr:g}, moved {still} "
                    f"by {moved:.3f}: {mover}'s step reached {still}, or {still}'s took a learning rate not its own"
                )
            readings = after
        return problems

    async def u2(self) -> list[str]:
        apart = _apart(await self.readings("before the sampler saves"))
        if not min(apart.values()) > MARGIN * self.tolerance:
            return [
                f"the adapters' weights set them apart by only {_apart_shown(apart)}, too little to check their "
                f"samplers (raise learning_rate)"
            ]
        problems = []
        paths = {}
        for name, client in self.clients.items():
            self.doing = f"a sampler save of {name}"
            paths[name] = await _saved(client.save_weights_for_sampler_async(SAVE_NAME))
        if paths["A"] == paths["B"]:
            problems.append(f"the sampler saves of A and B under one name are one path, {paths['A']}")
        saved, published = {}, {}
        for name, path in paths.items():
            self.doing = f"a sampler of {name}'s sampler save"
            saved[name] = await self.session_of[name].create_sampling_client_async(model_path=path)
        for name, client in self.clients.items():
            self.doing = f"publishing {name}'s sampler"
            published[name] = await client.save_weights_and_get_sampling_client_async()
        for kind, samplers, through in (
            ("the samplers of their sampler saves", saved, "through the sampler of its sampler save"),
            ("the samplers they publish", published, "through the sampler it publishes"),
        ):
            served = _apart(
                {name: await self.reading(name, sampler, through, sampler=True) for name, sampler in samplers.items()}
            )
            if not all(served[name] > apart[name] / 2 for name in TOKENS):
                problems.append(
                    f"{kind} set A and B apart by {_apart_shown(served)}, where their weights do by "
                    f"{_apart_shown(apart)}: a sampler serves another adapter's weights, or the base model's"
                )
        return problems

    async def u4(self) -> list[str]:
        a, b = self.clients["A"], self.clients["B"]
        lr, tolerance = self.learning_rate, self.tolerance
        problems = []
        self.doing = "a step of A"
        await _step([(a, "A", 1.0, lr)])  # so that A's optimizer holds a gradient of A's tokens
        b_before = await self.reading("B", b, "before A's save")
        self.doing = "a save of A with its optimizer while B steps"
        stepping = await _submit([(b, "B", 1.0, lr)])
        path = await _saved(a.save_state_async(SAVE_NAME))
        await _results(stepping)
        saved = await self.reading("A", a, "after its save")
        self.doing = "a step of A with no gradient"
        await _step([(a, "A", 0.0, lr)])
        momentum = (await self.reading("A", a, "after its step with no gradient"))["A"] - saved["A"]
        if not (seen := momentum > MARGIN * tolerance):
            problems.append(
                f"a step of A with no gradient moved its tokens by {momentum:+.3f}, too little to see its optimizer's "
                f"state: the service drops the momentum of such a step, or steps at learning rate {lr:g} are too "
                f"small (raise learning_rate)"
            )
        self.doing = "reopening A's save with its optimizer, in a session of its own, while B steps"
        stepping = await _submit([(b, "B", 1.0, lr)])
        self.reopened = reopened = await self.session().create_training_client_from_state_with_optimizer_async(path)
        await _results(stepping)
        if _distance(restored := await self.reading("A reopened", reopened), saved) > tolerance:
            problems.append(f"A reopened gives {_given(restored)}, where A gave {_given(saved)} at its save")
        self.doing = "a step of A reopened, with no gradient"
        await _step([(reopened, "A", 0.0, lr)])
        reopened_after = await self.reading("A reopened", reopened, "after its step with no gradient")
        reopened_momentum = reopened_after["A"] - restored["A"]
        if seen and abs(reopened_momentum - momentum) > momentum / 2:
            problems.append(
                f"a step with no gradient moved A reopened by {reopened_momentum:+.3f}, and A by {momentum:+.3f}: "
                "A did not reopen with its optimizer's state (its moments and step count)"
            )
        b_after = await self.reading("B", b, "after its steps beside A's save and reopening")
        if not (rose := b_after["B"] - b_before["B"]) > tolerance:
            problems.append(f"B's steps beside A's save and reopening moved B's tokens by {rose:+.3f}: B lost them")
        return problems

    async def u3(self) -> list[str]:
        b = self.clients["B"]
        others = {"B": (b, "B"), "A reopened": (self.reopened, "A")}
        problems = []
        self.doing = "a sampler save of B"
        path = await _saved(b.save_weights_for_sampler_async(f"{SAVE_NAME}-b"))
        self.doing = "a sampler of B's sampler save"
        sampler = await self.session_of["B"].create_sampling_client_async(model_path=path)
        before = {}
        for name, (client, _) in others.items():
            before[name] = await self.reading(name, client, "before A's session ends")
        served = await self.reading("B", sampler, "through its sampler before A's session ends", sampler=True)
        if not await self.end_a():
            return [
                "not checked: A's model still answers after its session's end event, so this service does not end a "
                "session on it"
            ]
        for name, (client, _) in others.items():
            moved = _distance(await self.reading(name, client, "after A's session ended"), before[name])
            if moved > self.tolerance:
                problems.append(f"{name}'s weights moved by {moved:.3f} as A's session ended")
        after = await self.reading("B", sampler, "through its sampler after A's session ended", sampler=True)
        if (moved := _distance(after, served)) > self.tolerance:
            problems.append(f"B's sampler serves weights {moved:.3f} from those it served before A's session ended")
        for name, (client, tokens) in others.items():
            self.doing = f"a step of {name} after A's session ended"
            await _step([(client, tokens, 1.0, self.learning_rate)])
        self.doing = "a sampler save of B after A's session ended"
        await _saved(b.save_weights_for_sampler_async(f"{SAVE_NAME}-b-after"))
        return problems

    async def end_a(self) -> bool:
        """End A's session as the Tinker SDK ends one after a fatal error: its session-end event, then no heartbeat.
        Whether it ended: A's model no longer answers after the event.  A is asked before the heartbeat stops, since
        the SDK's telemetry stops with it, and the read that fails sends its own session-end event through it."""

        session = self.session_of["A"]
        try:
            self.doing = "ending A's session"
            await _end_event(session)
            self.doing = "reading A after its session's end event"
            return not await _answers(self.clients["A"])
        finally:
            self.close(session)


def _datum(name: str, weight: float) -> Any:
    """A row that trains on adapter ``name``'s tokens (:data:`TOKENS`), each at loss weight ``weight``."""

    import tinker

    tokens = TOKENS[name]
    count = len(tokens) - 1
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(tokens[:-1])),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=list(tokens[1:]), dtype="int64", shape=[count]),
            "weights": tinker.TensorData(data=[weight] * count, dtype="float32", shape=[count]),
        },
    )


async def _read(client: Any) -> dict[str, float]:
    """The mean logprob that a training client's weights give each adapter's tokens (``forward``: no gradient)."""

    future = await client.forward_async([_datum(name, 1.0) for name in TOKENS], "cross_entropy")
    outputs = (await future.result_async()).loss_fn_outputs
    return {name: _mean(name, output["logprobs"].data) for name, output in zip(TOKENS, outputs, strict=True)}


async def _served(sampler: Any) -> dict[str, float]:
    """The mean logprob that a sampling client gives each adapter's tokens (the first token has none)."""

    import tinker

    return {
        name: _mean(name, (await sampler.compute_logprobs_async(tinker.ModelInput.from_ints(list(tokens))))[1:])
        for name, tokens in TOKENS.items()
    }


async def _answers(client: Any) -> bool:
    """Whether a training client's model answers a ``forward``."""

    try:
        await (await client.forward_async([_datum("A", 1.0)], "cross_entropy")).result_async()
    except Exception:  # noqa: BLE001 - whatever the error, the model did not answer
        return False
    return True


async def _submit(steps: Sequence[_Step]) -> list[Any]:
    """Submit an optimizer step of each client: every ``forward_backward``, then every ``optim_step``, before any
    result is awaited, as a step of several policies submits them; their futures."""

    import tinker

    futures = []
    for client, tokens, weight, _ in steps:
        futures.append(await client.forward_backward_async([_datum(tokens, weight)], "cross_entropy"))
    for client, _, _, learning_rate in steps:
        futures.append(await client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate)))
    return futures


async def _results(futures: Sequence[Any]) -> None:
    for future in futures:
        await future.result_async()


async def _step(steps: Sequence[_Step]) -> None:
    await _results(await _submit(steps))


async def _saved(save: Any) -> str:
    """The path that a save (``save_state_async`` or ``save_weights_for_sampler_async``, not yet awaited) wrote."""

    return (await (await save).result_async()).path


async def _end_event(session: Any) -> None:
    """Send ``session``'s session-end event, as the Tinker SDK sends it after a fatal error (its telemetry's
    ``log_fatal_exception``, which waits until the event is sent); a self-hosted service may release the session's
    models at once on it."""

    if (telemetry := session.get_telemetry()) is not None:
        await telemetry.log_fatal_exception(RuntimeError(_ENDING))


def _apart(readings: dict[str, dict[str, float]]) -> dict[str, float]:
    """How much more each adapter's reading gives its own tokens than the other adapter's reading gives them."""

    return {"A": readings["A"]["A"] - readings["B"]["A"], "B": readings["B"]["B"] - readings["A"]["B"]}


def _distance(reading: dict[str, float], other: dict[str, float]) -> float:
    return max(abs(reading[name] - other[name]) for name in TOKENS)


def _mean(name: str, values: Sequence[Any]) -> float:
    """The mean of the logprobs that a reading gives adapter ``name``'s tokens; ``ValueError`` when it is not a finite
    number (weights that read NaN pass every comparison)."""

    mean = sum(float(value) for value in values) / len(values)
    if not math.isfinite(mean):
        raise ValueError(f"the mean logprob of {name}'s tokens is {mean}, not a finite number")
    return mean


def _apart_shown(apart: dict[str, float]) -> str:
    return f"{apart['A']:+.3f} on A's tokens and {apart['B']:+.3f} on B's"


def _given(reading: dict[str, float]) -> str:
    return f"{reading['A']:.3f} to A's tokens and {reading['B']:.3f} to B's"


def _error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
