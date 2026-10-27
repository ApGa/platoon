"""Step hooks (RFC 0021, section 9): work at the end of each step of any trainer, after the step's own updates and
its commit, before the step publishes.

A hook (:class:`StepHook`; :class:`StepHookBase` gives every method a default) registers with the run in the trainer's
``check()`` or ``prepare()`` (``run.hooks.register``), which also makes one with state to resume (``keys``) a
participant of the run's checkpoints.  The rl trainer also attaches the hooks its config names
(``trainer_options.hooks``: :mod:`platoon.train.trainer.hook_config`).  At each step's end one sequence runs
(:func:`finish_step`), in the rl trainer's step and in ``run.end_step`` alike:

1. each hook's ``drain``, in registration order: side work that samples the policy finishes or stops;
2. the step is open (``clock.step_open``), and each hook's ``update`` runs its optimizer updates through
   ``end.train(...)`` or ``end.learner(...)``, on the policy's training client and Adam state;
3. each policy that the step or a hook updated publishes once: one save and one new version, however many updates
   (platoon counts the hooks' updates: ``end.hook_updates``);
4. each hook hears what was published (``published``), also when nothing was; the run also tells them of the launch's
   first samplers (:func:`hear_launch`);
5. the clock advances.

With no hook registered, steps 1, 2 and 4 are skipped, not awaited empty, and the step's end is not even built: the
step publishes and advances as it always did.  A hook that raises in ``drain`` or ``update`` fails the start before
anything publishes, as a failed update does.  A hook that is not ``critical`` only hears publications, and its errors
there are logged while training goes on.  An update left open, sent and not applied, refuses the step's end, with
hooks or without.  From the hooks' drain to the advance the clock says the step is ending (``clock.ending``): when
the trainer's steps are not atomic, an interrupt then saves no checkpoint of the step either (rule 10), since its own
updates and its commit are done and its hooks' perhaps half.

Imports no Tinker, so that ``platoon.experimental`` re-exports it.
"""

import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Self, cast

from platoon.train.policy.spec import PolicyId

if TYPE_CHECKING:
    from platoon.train.checkpoints.participants import CheckpointParticipant, RunCheckpoints
    from platoon.train.objectives.api import Objective
    from platoon.train.policy.learner import Learner
    from platoon.train.policy.policy import Policy
    from platoon.train.trainer.run import TrainerRun
    from platoon.utils.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)

# A training draw's key: its start's number and its number among that start's draws, never reused in a run.
DrawKey = tuple[int, int]


class StepHook(Protocol):
    """Work at the end of each step of any trainer, after the step's own updates and its commit, before it publishes.
    A class of this protocol defines all four methods; subclass :class:`StepHookBase`, whose methods are empty, to
    leave some out.  ``critical`` is optional (True when left out).  A hook with state to resume is also a
    ``CheckpointParticipant``.  Register it in the trainer's ``check()`` or ``prepare()``."""

    # Its log lines, its waits and its metrics' prefix (``<name>/``); no two hooks of a run share one.
    name: str

    async def drain(self, end: "StepEnd") -> None:
        """Finish or stop the side work that samples the current sampler, before the step's hooks update the policy
        and it publishes.  Raising fails the start before anything publishes."""
        ...

    async def update(self, end: "StepEnd") -> None:
        """Optimizer updates of the run's policies, through ``end.train(...)`` or ``end.learner(...)`` only.  Raising
        fails the start before anything publishes."""
        ...

    async def published(self, end: "StepEnd") -> None:
        """At every step's end, after it published (``end.published``, empty when nothing did); and once after the
        launch's first sampler (``end.kind`` ``"launch"``)."""
        ...

    async def aclose(self) -> None:
        """At the start's end, also after a failure: close what the hook opened."""
        ...


class StepHookBase:
    """An optional base of step hooks: every method empty, critical, and :meth:`from_run`, which builds a hook that
    the config names."""

    name: str
    # False: the hook only hears publications (``published``, ``aclose``), and an error it raises there is logged
    # while training goes on; it may not drain or update.
    critical: ClassVar[bool] = True

    @classmethod
    def from_run(cls, run: "TrainerRun", options: Mapping[str, Any]) -> Self:
        """The hook that the config attaches (``trainer_options.hooks``), built as the run checks its trainer, before
        any service opens, with the options the config gives it: by default ``cls()``, which takes none."""

        if options:
            raise ValueError(f"{cls.__qualname__} takes no options; give it from_run(run, options) to take some")
        return cls()

    async def drain(self, end: "StepEnd") -> None:
        """Nothing to drain."""

    async def update(self, end: "StepEnd") -> None:
        """No update."""

    async def published(self, end: "StepEnd") -> None:
        """Nothing to hear."""

    async def aclose(self) -> None:
        """Nothing to close."""


@dataclass(frozen=True)
class RunSide:
    """The run's side of a step's end, which its hooks do not use: the policies a hook may update, whose optimizer
    steps platoon counts from the step's end on; the learner of each, which a hook's learner shares the training
    client, the waits and the checks of; the run's waits (``run.wait``); and whether a hook's wait may stop the step
    (``interruptible``: the trainer's steps are atomic, rule 10)."""

    policies: Mapping[PolicyId, "Policy"]
    learner_of: Callable[[PolicyId], "Learner"]
    wait: Callable[..., Awaitable[Any]]
    interruptible: bool


@dataclass(eq=False)
class StepEnd:
    """What the end of a step tells its hooks.  ``hook_updates`` is platoon's own count of the hooks' updates: the
    optimizer steps each policy took since the step's end began, whatever a hook counts.  ``train``, ``learner`` and
    ``wait`` are the run's; a test builds a step's end with :meth:`for_test`."""

    # "step": the end of a step; "launch": the start's first samplers, which the hooks hear once (published()).
    kind: Literal["launch", "step"]
    # The step that ends (the clock's step before it advances); at launch, the start's first step.
    step: int
    # The step's own optimizer updates, by policy, before any hook's (rl: its policy-gradient updates).
    step_updates: Mapping[PolicyId, int]
    # The draws the step committed, by key, each with its fate (consumed, stale or failed).
    consumed: Mapping[DrawKey, str]
    # The step's metrics row: what a hook records here is logged with the step.
    tracker: "StatsTracker"
    # Who published at the end of this step, at which new version; set before the hooks hear it.
    published: Mapping[PolicyId, int] = field(default_factory=dict)
    # The run's side (None: a step's end that a test built).
    _run: RunSide | None = field(default=None, kw_only=True, repr=False)
    # The hook whose update() runs, which alone may update meanwhile.
    _updating: StepHook | None = field(default=None, init=False, repr=False)
    # Whether the step's end is over (its hooks heard it): a learner of it belongs to an earlier step from then on.
    _over: bool = field(default=False, init=False, repr=False)
    # Each policy's optimizer steps as the step's end began.
    _steps_before: dict[PolicyId, int] = field(default_factory=dict, init=False, repr=False)
    # What a test's step's end hands out as every learner (for_test).
    _test_learner: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._run is not None:
            self._steps_before = {policy: trained.optimizer_steps for policy, trained in self._run.policies.items()}

    @classmethod
    def for_test(
        cls,
        *,
        kind: Literal["launch", "step"] = "step",
        step: int = 0,
        step_updates: Mapping[PolicyId, int] | None = None,
        consumed: Mapping[DrawKey, str] | None = None,
        tracker: "StatsTracker | None" = None,
        published: Mapping[PolicyId, int] | None = None,
        learner: Any = None,
    ) -> "StepEnd":
        """A step's end built by hand, to test a hook alone: ``end.learner(...)``, and so ``end.train(...)``, hand
        out ``learner`` whatever the policy and the objective (a fake of
        :class:`~platoon.train.policy.learner.Learner` that records what the hook sends; None: they raise),
        ``end.wait`` awaits, and ``hook_updates`` is empty.  ``tracker`` defaults to a fresh ``StatsTracker``."""

        from platoon.utils.stats_tracker import StatsTracker

        end = cls(
            kind=kind,
            step=step,
            step_updates=step_updates or {},
            consumed=consumed or {},
            tracker=tracker if tracker is not None else StatsTracker(),
            published=published or {},
        )
        end._test_learner = learner
        return end

    @property
    def hook_updates(self) -> dict[PolicyId, int]:
        """The hooks' optimizer updates so far, by policy, as platoon counts them."""

        if self._run is None:
            return {}
        return {
            policy: trained.optimizer_steps - self._steps_before[policy]
            for policy, trained in self._run.policies.items()
        }

    def learner(self, policy: PolicyId, *, objective: "Objective") -> "Learner":
        """The learner of ``policy`` under ``objective``, for the hook whose ``update()`` runs, which may use it there
        only (it refuses to send anything after, in a later step too, and so does a learner made from it under another
        objective): the policy's training client and optimizer parameters, the ones the step's own updates used,
        failing closed when the trainer's needs say so, its metrics under ``<hook name>/`` in the step's row."""

        run, hook = self._run, self._updating
        if run is None:
            if self._test_learner is None:
                raise RuntimeError("this step's end was built by hand: StepEnd.for_test(learner=...) gives a learner")
            return self._test_learner
        if hook is None:
            raise RuntimeError("end.learner(...) is a step hook's, during its update() only")
        if policy not in run.policies:
            raise ValueError(f"policy {policy!r} trains no update at this step's end: {sorted(run.policies)} do")

        def guard() -> None:
            if self._updating is hook:
                return
            if self._over:
                raise RuntimeError(
                    f"step hook {hook.name!r} sends through a learner of an earlier step, step {self.step}: a learner "
                    "updates the policy during the update() that took it only, so take one in each update()"
                )
            raise RuntimeError(f"step hook {hook.name!r} updates the policy during its update() only")

        return run.learner_of(policy).for_objective(objective, metric_prefix=f"{hook.name}/", guard=guard)

    async def train(
        self, policy: PolicyId, datums: Sequence[Any], *, objective: "Objective", adam_params: Any = None
    ) -> None:
        """One optimizer update of ``policy`` on ``datums`` (``tinker.Datum``, sent as given) under ``objective``:
        one forward_backward of them all, then one optimizer step, with ``adam_params`` (a ``tinker.AdamParams``) for
        this update alone (None: the policy's, ``train.optimizer``'s).  ``end.learner`` sends several microbatches
        into one update."""

        update = self.learner(policy, objective=objective).update()
        await update.submit(list(datums), list(datums), minibatch=0, microbatch=0)
        await update.apply(adam_params=adam_params)

    async def wait(
        self, awaitable: Awaitable[Any], *, name: str, timeout: float | None = None, interruptible: bool = False
    ) -> Any:
        """Await ``awaitable`` as the run's waits do (``run.wait``): a log line and a watchdog heartbeat every minute,
        for at most ``timeout`` seconds.  With ``interruptible``, ``TrainInterrupted`` once a stop is asked for, when
        the trainer's steps are atomic (``needs().atomic_steps``: the run then saves nothing of the step).  When they
        are not (the rl trainer's), a stop request does not end the wait, and the hook cannot see it: the wait runs to
        its end, or to ``timeout`` (``TimeoutError``, which fails the start), so bound what may not end by itself; the
        stop takes effect at the next step's boundary, so that no checkpoint saves half a step."""

        if self._run is None:
            return await awaitable
        interruptible = interruptible and self._run.interruptible
        return await self._run.wait(awaitable, name=name, timeout=timeout, interruptible=interruptible)


class StepHooks:
    """The run's step hooks, in the order they registered, which is the order they run in.  A hook with ``keys`` is
    a participant of the run's checkpoints (``checkpoints``) too, which registering it here registers it with;
    ``joined`` hears of each hook, so that the run closes it."""

    def __init__(
        self, checkpoints: "RunCheckpoints | None" = None, *, joined: Callable[[Any], None] | None = None
    ) -> None:
        self._hooks: list[StepHook] = []
        self._checkpoints = checkpoints
        self._joined = joined

    def register(self, hook: StepHook) -> None:
        """``hook`` runs at every step's end from now on, after those registered before it; one with ``keys`` takes
        part in every full checkpoint too (``run.checkpoints.register``).  Refused (ValueError): a hook without a
        name, a name another hook has, a hook that is not critical but drains or updates; and (``ConfigError``, exit
        78) keys that are not its own, the draws' among them."""

        name = getattr(hook, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"a step hook has a name, its metrics' prefix: {hook!r} has none")
        if any(other.name == name for other in self._hooks):
            raise ValueError(f"two step hooks are named {name!r}: each needs a name of its own")
        if not is_critical(hook) and (_defines(hook, "drain") or _defines(hook, "update")):
            raise ValueError(
                f"step hook {name!r} is not critical, so it only hears publications: it may not drain or update"
            )
        if getattr(hook, "keys", ()) and self._checkpoints is not None:
            self._checkpoints.register(cast("CheckpointParticipant", hook), hook=True)
        self._hooks.append(hook)
        if self._joined is not None:
            self._joined(hook)

    def __iter__(self) -> Iterator[StepHook]:
        return iter(self._hooks)

    def __len__(self) -> int:
        return len(self._hooks)


class _Clock(Protocol):
    """The clock a step's end opens and advances (:class:`~platoon.train.trainer.run.StepClock`): whether the step
    sent training work, whether its end runs its hooks, and each policy's update that is open, sent and not applied."""

    step_open: bool
    ending: bool
    open_updates: dict[PolicyId, Any]

    def advance(self, *, consumed: Any = None) -> None: ...


async def finish_step(
    hooks: StepHooks,
    end_of: Callable[[], StepEnd],
    *,
    clock: _Clock,
    source: Any,
    publish: Callable[[StepEnd | None], Awaitable[Mapping[PolicyId, int]]],
) -> Mapping[PolicyId, int]:
    """A step's end, after its own updates and its commit: with hooks, the step's end (``end_of()``) is told to each:
    they drain, the step is open while they update, ``publish(end)`` publishes each policy the step or a hook updated,
    once, and the hooks hear who published at which version, the clock saying meanwhile that the step is ending.  Then
    ``clock`` advances, moving on the horizon of ``source``, what the step drew its groups from.  With no hook,
    ``publish(None)`` (the step's own updates alone) then the advance, nothing else.  An update left open refuses the
    step's end, before the hooks run and after they update.  Who published, at which version."""

    if clock.open_updates:
        raise _left_open(clock, "the step")
    if not hooks:
        published = await publish(None)
    else:
        clock.ending = True
        end = end_of()
        await _drain(hooks, end)
        clock.step_open = True
        await _update(hooks, end)
        if clock.open_updates:
            raise _left_open(clock, "a step hook")
        end.tracker.scalar(**{"batch/hook_optimizer_updates": float(sum(end.hook_updates.values()))})
        end.published = published = await publish(end)
        await _hear(hooks, end)
        end._over = True
    clock.advance(consumed=source)
    return published


async def hear_launch(hooks: StepHooks, end: StepEnd) -> None:
    """The launch's first samplers (``end``, of kind ``"launch"``), which each hook hears once (``published``)."""

    await _hear(hooks, end)


def _left_open(clock: _Clock, who: str) -> RuntimeError:
    """What refuses a step's end that finds an update open, sent by ``who`` and not applied (its optimizer step not
    sent, or not counted yet): the next optimizer step would take in its gradients, or the publication would carry an
    optimizer step that no count holds."""

    policies = ", ".join(repr(policy) for policy in sorted(clock.open_updates))
    return RuntimeError(
        f"{who} left an update of policy {policies} open, sent but not applied (submit without apply, or an apply "
        "still running): every update applies before the step's hooks update or the step publishes"
    )


async def _drain(hooks: StepHooks, end: StepEnd) -> None:
    """Each hook's ``drain``, in registration order, under the run's waits (``drain <name>``)."""

    for hook in hooks:
        if (drain := getattr(hook, "drain", None)) is not None:
            await end.wait(drain(end), name=f"drain {hook.name}")


async def _update(hooks: StepHooks, end: StepEnd) -> None:
    """Each hook's ``update``, in registration order, which alone may update meanwhile."""

    for hook in hooks:
        if (update := getattr(hook, "update", None)) is None:
            continue
        end._updating = hook
        try:
            await update(end)
        finally:
            end._updating = None


async def _hear(hooks: StepHooks, end: StepEnd) -> None:
    """Each hook's ``published``, in registration order: a critical hook's error fails the start, another's is
    logged."""

    for hook in hooks:
        if (published := getattr(hook, "published", None)) is None:
            continue
        if is_critical(hook):
            await published(end)
            continue
        try:
            await published(end)
        except Exception:
            logger.exception(
                f"Step hook {hook.name} failed to hear the end of step {end.step}; it is not critical, so training "
                "goes on"
            )


def is_critical(hook: Any) -> bool:
    """Whether an error of ``hook`` (a step hook or a checkpoint participant) fails the start: unless it says it is
    not critical."""

    return bool(getattr(hook, "critical", True))


def _defines(hook: Any, method: str) -> bool:
    """Whether ``hook`` has a ``method`` of its own (not :class:`StepHookBase`'s empty one)."""

    own = getattr(type(hook), method, None)
    return own is not None and own is not getattr(StepHookBase, method)
