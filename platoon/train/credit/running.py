"""``credit: running``: each scored agent's rows credited against running statistics of the groups that settled before
it sampled, for an entry whose groups have too few members to center on each other, such as one rollout each.  It is
REINFORCE++'s baseline, scale and clip with the step's batch replaced by decayed statistics, since an entry's groups
settle apart and are credited one at a time (RFC 0001's ``rae``, a baseline per role, under this name).

For each agent of the group with a reward ``R`` and rows, on every action token of its rows::

    z = clip((R - b) / sigma, -clip, clip), and 0 when R < 0 and z > 0 (cap_penalties: a penalty never pays)

- **Per role and policy.** The agents of each role (``AgentRun.role``: ``policy`` for a root) of each policy have
  statistics of their own, which age in that policy's versions: a root and its subagents are never centered on each
  other's rewards.
- **Strata.** The agent's stratum is the task's values of the columns ``strata`` names, in order
  (``CreditGroup.columns``: a Harbor task's index fields, a dataset loader's columns).  Under ``[kind, mode]`` a task
  of kind ``masked`` and mode ``numeric`` is in the stratum ``(masked, numeric)``, below ``(masked)``, below the root
  ``()`` that holds every sample.  A task without one of the columns, or with an empty value, is credited at the
  stratum above it (warned once per column).
- **The baseline** ``b`` of a stratum is its decayed mean reward shrunk toward its parent's, ``(S + shrink *
  b_parent) / (W + shrink)``, with ``S`` and ``W`` the decayed sums of its samples' rewards and weights; the root's is
  ``S / W``, and 0 while it has no weight.  **The scale** ``sigma`` is the decayed root mean square of the settled
  samples' residuals, each its reward less the baseline it was credited against, and at least :data:`MIN_SCALE`.
- **Decay.** A sample enters the statistics at full weight when its group settles, and every statistic halves each
  ``half_life`` versions of its policy, the scale's each twice as many, so that each outcome counts at the share it
  settles at, however long its rollout took.
- **The read.** ``b`` and ``sigma`` are read from the statistics as they were when the agent's policy reached the
  version the agent sampled at, the oldest of its rows' (``TrainingSample.version``): only what settled before it
  began sampling, so that how long its own group took to settle changes nothing of its baseline.  They are kept as
  far back as the policy's rows of the entry may be old and still train (its staleness bound,
  ``CreditSetup.max_staleness``), counted from the version of the last settle: an agent that sampled earlier, whose
  oldest row the stale filter drops, trains at 0, and settles.  So the entry needs a bound
  (:meth:`Running.check`); without one the setup gives (outside a run), :data:`DEFAULT_WINDOW` versions are kept.
- **Priors and the warm-up.** ``priors`` seed each role's statistics, as ``weight`` samples at each prior mean (the
  root's, and that of each stratum they name) and ``weight`` residuals of their ``scale``, which decay as samples do.
  Without priors every agent trains at 0 until the statistics it reads hold :data:`WARMUP_SAMPLES` settled samples;
  its own sample still settles.

An agent without a reward (interrupted, or a child whose verifier gave no verdict) trains nothing and moves nothing,
and so does one whose reward is not a finite number (warned of once), which would spoil every baseline it entered; a
scored agent without rows (sampled out) settles.  The statistics move only in ``settle``, as the run files each
group, and are the estimator's state: for each role of each policy, the oldest statistics a read may still need and
every settle since, from which ``load_state`` rebuilds the rest exactly.  A full checkpoint saves that with the groups
it keeps, about one copy of the statistics and the settles of the bound's versions, so a resume credits as the run
would have; a stratum whose weight became negligible leaves them (:data:`NEGLIGIBLE_WEIGHT`).  Each group's metrics:
``credit/running/advantage/avg`` (the mean advantage, which drifts from 0 when the baselines lag),
``credit/agent_advantage/abs_avg``, ``credit/running/baseline/avg``, ``credit/running/scale/avg``,
``credit/running/age/avg`` (the policy versions between sampling and credit), and the shares of the agents in the
warm-up, whose statistics were no longer kept, clipped, and capped (``credit/running/warmup``,
``.../out_of_ring``, ``.../clipped``, ``.../capped``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from platoon.train.credit.api import AgentRun, CreditCheckContext, CreditEstimatorBase, CreditGroup, CreditSetup
from platoon.train.labels import label_text
from platoon.train.policy.spec import PolicyId

logger = logging.getLogger(__name__)

# How many versions back a read reaches for a policy whose rows have no bound the setup gives: a setup built outside a
# run, or an entry without a bound, which the load-time check refuses (Running.check).
DEFAULT_WINDOW = 64
# Without priors, the settled samples the statistics an agent reads must hold before it trains on them.
WARMUP_SAMPLES = 32
# The least scale an advantage is divided by.
MIN_SCALE = 0.1
# A stratum whose decayed weight falls below this, about 20 half-lives after its last sample, leaves the statistics,
# and reads its parent's baseline after, from which its own then differed by less than a millionth of their gap: the
# statistics hold the strata of recent samples only.
NEGLIGIBLE_WEIGHT = 1e-6
# What a stratum is: the task's values of the strata columns, in order (``()``: the root, every sample).
Stratum = tuple[str, ...]
# One settled sample, as the state keeps it: the policy version it settled at, its stratum, its reward and its residual.
Settle = tuple[int, Stratum, float, float]


@dataclass(frozen=True)
class StratumPrior:
    """One stratum's prior mean reward, under ``credit_options.priors.strata``."""

    # The stratum: its values of the strata columns, in order, as many as one or more of them ([masked], or [masked,
    # numeric] for the stratum below it), each as the task's column holds it (a string, a number or a boolean: 3 is
    # the column's 3, true its true).
    stratum: list[str | int | float | bool]
    # Its prior mean reward.
    mean: float

    def __post_init__(self) -> None:
        values = self.stratum if isinstance(self.stratum, list | tuple) else None
        texts = [label_text(value) for value in values] if values else []
        if not texts or None in texts:
            raise ValueError(
                f"stratum: one or more values of the strata columns, in order, each a non-empty string, a number or "
                f"a boolean, got {self.stratum!r}"
            )
        object.__setattr__(self, "stratum", texts)  # as the task's columns read (CreditGroup.columns)


@dataclass(frozen=True)
class RunningPriors:
    """``credit_options.priors`` under ``credit: running``: where the statistics start, such as a measurement of the
    policy before training.  :class:`RunningOptions` checks them (:func:`priors_problem`)."""

    # The prior mean reward of every sample: the root stratum's.
    mean: float
    # The prior root mean square of the residuals: the scale the first advantages are divided by.
    scale: float
    # How many samples each prior weighs as; it decays as samples do.
    weight: float = 16.0
    # The prior mean rewards of strata, each once.
    strata: list[StratumPrior] = field(default_factory=list)


@dataclass(frozen=True)
class RunningOptions:
    """``credit_options`` under ``credit: running``."""

    # The options that name files, which the config loader makes absolute against the config file's directory
    # (platoon.utils.config; the credit registry's CORE_PATH_OPTIONS says so without importing this module).
    path_options: ClassVar[tuple[str, ...]] = ("priors",)

    # The task columns the baselines are kept by, coarsest first, each once: columns of the entry's tasks, such as a
    # Harbor task's index fields or a dataset loader's columns (not task_id, nor a column of platoon's own).  Empty: one
    # baseline for every sample.
    strata: list[str] = field(default_factory=list)
    # Where the statistics start: the priors (mean, scale, weight, strata), or the path of a YAML or JSON file that
    # holds them, relative to the config file.  None: none, and an agent trains at 0 until the statistics it reads hold
    # 32 settled samples.
    priors: RunningPriors | str | None = None
    # How many policy versions a settled sample's weight in the baselines takes to halve; in the scale, twice as many.
    half_life: float = 4.0
    # How many samples a stratum's parent's baseline weighs as in the stratum's (0: none, a stratum's own mean).
    shrink: float = 8.0
    # The largest advantage, in either sign.
    clip: float = 3.0
    # A negative reward never gets a positive advantage, whatever its stratum's baseline.
    cap_penalties: bool = True

    def __post_init__(self) -> None:
        from platoon.train.components import platoon_column

        strata = self.strata
        if not isinstance(strata, list) or not all(isinstance(column, str) and column for column in strata):
            raise ValueError(f"strata: a list of task columns, got {strata!r}")
        if len(set(strata)) != len(strata):
            raise ValueError(f"strata: each column once, got {strata!r}")
        if own := [column for column in strata if platoon_column(column)]:
            raise ValueError(f"strata: {', '.join(own)} is platoon's own column, not one of the task's")
        _finite("half_life", self.half_life, above=0.0)
        _finite("shrink", self.shrink, at_least=0.0)
        if isinstance(self.clip, bool) or not isinstance(self.clip, int | float) or not self.clip > 0:
            raise ValueError(f"clip: a number above 0, got {self.clip!r}")
        if not isinstance(self.cap_penalties, bool):
            raise ValueError(f"cap_penalties: true or false, got {self.cap_penalties!r}")
        priors, source = self.priors, None
        if isinstance(priors, str):
            if not priors:
                raise ValueError("priors: the priors, or the path of a file that holds them, got ''")
            priors, source = read_priors(priors), priors
            object.__setattr__(self, "priors", priors)
        elif priors is not None and not isinstance(priors, RunningPriors):
            raise ValueError(f"priors: the priors, or the path of a file that holds them, got {priors!r}")
        if priors is not None and (problem := priors_problem(priors, strata)) is not None:
            raise ValueError(f"priors: {source}: {problem}" if source is not None else f"priors: {problem}")


def priors_problem(priors: RunningPriors, strata: list[str]) -> str | None:
    """Why ``priors`` cannot seed the statistics of ``strata``, under their own keys (``scale: ...``), or None."""

    for name, above in (("mean", None), ("scale", 0.0), ("weight", 0.0)):
        if (problem := _number_problem(name, getattr(priors, name), above=above)) is not None:
            return problem
    named: list[tuple[str, ...]] = []
    for index, prior in enumerate(priors.strata):
        where = f"strata.{index}"
        if len(prior.stratum) > len(strata):
            columns = ", ".join(strata) or "none"
            return f"{where}.stratum has {len(prior.stratum)} value(s), more than strata names columns ({columns})"
        if (problem := _number_problem(f"{where}.mean", prior.mean)) is not None:
            return problem
        named.append(tuple(str(value) for value in prior.stratum))
    if len(set(named)) != len(named):
        return f"strata: each stratum once, got {[list(stratum) for stratum in named]!r}"
    return None


def read_priors(path: str | Path) -> RunningPriors:
    """The priors a YAML or JSON file holds (``credit_options.priors`` as a path); a ``ValueError`` under ``priors``
    that says why for a file that cannot be read or holds no priors (its values :func:`priors_problem` checks)."""

    import yaml

    from platoon.options import schema_options

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"priors: {path} cannot be read ({exc.strerror or exc})") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"priors: {path} is not YAML or JSON ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError(f"priors: {path} holds {type(data).__name__}, not a mapping of priors (mean, scale, ...)")
    parsed, problems = schema_options(RunningPriors, name="priors").check(data, "")
    if problems:
        raise ValueError(f"priors: {path}: {'; '.join(problems)}")
    return parsed


def _finite(name: str, value: Any, *, above: float | None = None, at_least: float | None = None) -> None:
    """Refuse ``value`` of option ``name`` unless it is a finite number, above ``above`` and at least ``at_least``."""

    if (problem := _number_problem(name, value, above=above, at_least=at_least)) is not None:
        raise ValueError(problem)


def _number_problem(name: str, value: Any, *, above: float | None = None, at_least: float | None = None) -> str | None:
    """Why ``value`` of ``name`` is not a finite number above ``above`` and at least ``at_least``, or None."""

    ok = not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
    if ok and (above is None or value > above) and (at_least is None or value >= at_least):
        return None
    bound = f" above {above:g}" if above is not None else f" of {at_least:g} or more" if at_least is not None else ""
    return f"{name}: a finite number{bound}, got {value!r}"


# --- the statistics ---------------------------------------------------------------------------------------------------


@dataclass
class _Statistics:
    """One role's decayed statistics at one version: for each stratum, the sums of its samples' rewards and weights;
    the sums of the residuals' squares and of their weights (the scale's); and how many samples settled into them."""

    sums: dict[Stratum, list[float]] = field(default_factory=dict)
    squares: list[float] = field(default_factory=lambda: [0.0, 0.0])
    count: int = 0

    @classmethod
    def seeded(cls, priors: RunningPriors | None) -> _Statistics:
        """The statistics before any sample settled: the priors' weight at each prior mean and residuals of their
        scale; nothing without priors."""

        if priors is None:
            return cls()
        weight = priors.weight
        sums = {(): [weight * priors.mean, weight]}
        sums.update({tuple(prior.stratum): [weight * prior.mean, weight] for prior in priors.strata})
        return cls(sums, [weight * priors.scale**2, weight])

    def decayed(self, versions: int, half_life: float) -> _Statistics:
        """A copy of the statistics ``versions`` versions on: the baselines' sums halved each ``half_life``, the
        scale's each twice that, without the strata (but the root) whose weight is now negligible."""

        kept, scale = 0.5 ** (versions / half_life), 0.5 ** (versions / (2 * half_life))
        sums = {
            stratum: [total * kept, weight * kept]
            for stratum, (total, weight) in self.sums.items()
            if not stratum or weight * kept >= NEGLIGIBLE_WEIGHT
        }
        return _Statistics(sums, [self.squares[0] * scale, self.squares[1] * scale], self.count)

    def baseline(self, stratum: Stratum, shrink: float) -> float:
        """The baseline of ``stratum``: each stratum's mean from the root down to it, shrunk toward its parent's."""

        total, weight = self.sums.get((), (0.0, 0.0))
        value = total / weight if weight > 0 else 0.0
        for depth in range(1, len(stratum) + 1):
            total, weight = self.sums.get(stratum[:depth], (0.0, 0.0))
            if weight + shrink > 0:
                value = (total + shrink * value) / (weight + shrink)
        return value

    def scale(self) -> float:
        """The scale an advantage is divided by: the residuals' root mean square, at least :data:`MIN_SCALE`."""

        total, weight = self.squares
        return max(math.sqrt(total / weight), MIN_SCALE) if weight > 0 else MIN_SCALE

    def add(self, stratum: Stratum, reward: float, residual: float) -> None:
        """Settle one sample at full weight: its reward into its stratum and each above it, its residual's square."""

        for depth in range(len(stratum) + 1):
            sums = self.sums.setdefault(stratum[:depth], [0.0, 0.0])
            sums[0] += reward
            sums[1] += 1.0
        self.squares[0] += residual * residual
        self.squares[1] += 1.0
        self.count += 1

    def to_json(self) -> dict[str, Any]:
        return {
            "sums": [[list(stratum), total, weight] for stratum, (total, weight) in sorted(self.sums.items())],
            "squares": list(self.squares),
            "count": self.count,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _Statistics:
        sums = {
            tuple(str(value) for value in stratum): [float(total), float(weight)]
            for stratum, total, weight in data["sums"]
        }
        total, weight = data["squares"]
        return cls(sums, [float(total), float(weight)], int(data["count"]))


@dataclass
class _Tracker:
    """The statistics of one role's agents of one policy, by version: as they were after the settles of each version
    from the oldest a read may still need (``base``, of ``base_version``; before anything left, the priors,
    ``initial``) to the newest, the live ones that take the settles of its version (``after``); and every settle since
    the oldest, in order (``settles``), from which they are rebuilt exactly (:meth:`from_json`).  A read reaches
    ``window`` versions back from the newest."""

    initial: _Statistics
    window: int
    half_life: float
    base: _Statistics | None = None
    base_version: int | None = None
    after: dict[int, _Statistics] = field(default_factory=dict)
    settles: list[Settle] = field(default_factory=list)

    @property
    def version(self) -> int | None:
        """The version of the newest settle; None before any."""

        return next(reversed(self.after)) if self.after else self.base_version

    def live(self) -> _Statistics:
        """The statistics with every settle so far, at the version of the newest."""

        if self.after:
            return self.after[next(reversed(self.after))]
        return self.base if self.base is not None else self.initial

    def at(self, version: int) -> _Statistics | None:
        """The statistics as they were when the policy's version reached ``version``: every settle of an earlier
        version, aged to it; None for a version the window no longer reaches."""

        newest = self.version
        if newest is None:
            return self.initial  # nothing settled
        if version <= newest - self.window or (self.base_version is not None and version <= self.base_version):
            return None
        earlier = [at for at in self.after if at < version]
        if earlier:
            at, statistics = earlier[-1], self.after[earlier[-1]]
        elif self.base is not None and self.base_version is not None:
            at, statistics = self.base_version, self.base
        else:
            return self.initial  # nothing settled before it
        return statistics.decayed(version - at, self.half_life)

    def now(self, version: int) -> _Statistics:
        """The statistics with every settle so far, as they are at ``version``, the policy's version now."""

        newest = self.version
        if newest is None or version <= newest:
            return self.live()
        return self.live().decayed(version - newest, self.half_life)

    def settle(self, version: int, stratum: Stratum, reward: float, residual: float) -> None:
        """Settle one sample at policy version ``version``: at a version past the newest, its statistics start from
        the newest's, aged to it, and those no read in the window needs any more leave (:meth:`_forget`); at an older
        one, which a resume took back, no time passed: it is the newest's."""

        newest = self.version
        if newest is None or version > newest or not self.after:
            at = version if newest is None else max(version, newest)
            self.after[at] = self.live().decayed(0 if newest is None else at - newest, self.half_life)
            self._forget(at)
            newest = at
        self.after[newest].add(stratum, reward, residual)
        self.settles.append((newest, stratum, reward, residual))

    def _forget(self, newest: int) -> None:
        """Let go of what the window that ends at ``newest`` no longer reaches: of the versions below it, the newest
        stays, as the base the reads after it start from, and the others leave with their settles."""

        below = [at for at in self.after if at <= newest - self.window]
        if not below:
            return
        self.base_version = below[-1]
        self.base = self.after[self.base_version]
        for at in below:
            del self.after[at]
        self.settles = [settle for settle in self.settles if settle[0] > self.base_version]

    def to_json(self) -> dict[str, Any]:
        return {
            "base_version": self.base_version,
            "base": self.base.to_json() if self.base is not None else None,
            "settles": [
                [version, list(stratum), reward, residual] for version, stratum, reward, residual in self.settles
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any], initial: _Statistics, window: int, half_life: float) -> _Tracker:
        """The tracker a state holds: its base, then each of its settles again, as they first ran (the same
        operations in the same order, so the statistics come out the same, to the bit)."""

        tracker = cls(initial, window, half_life)
        if data["base"] is not None:
            tracker.base, tracker.base_version = _Statistics.from_json(data["base"]), int(data["base_version"])
        for version, stratum, reward, residual in data["settles"]:
            tracker.settle(int(version), tuple(str(value) for value in stratum), float(reward), float(residual))
        return tracker


# --- the estimator ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Credit:
    """What an agent was credited with: its advantage, the baseline and scale it read (None: it read none), the policy
    versions between its sampling and its credit, and why its advantage is what it is."""

    advantage: float
    baseline: float | None
    scale: float | None
    age: int
    warmup: bool = False
    out_of_ring: bool = False
    clipped: bool = False
    capped: bool = False


class Running(CreditEstimatorBase):
    """Each scored agent's rows credited against running statistics of its role and stratum (the module's docstring)."""

    needs_agents = True
    credits_single_members = True
    options_schema = RunningOptions
    options: RunningOptions

    def __init__(self, setup: CreditSetup, options: RunningOptions | None = None) -> None:
        super().__init__(setup, options if options is not None else RunningOptions())
        priors = self.options.priors  # parsed: never a path
        self._initial = _Statistics.seeded(priors if isinstance(priors, RunningPriors) else None)
        self._trackers: dict[tuple[PolicyId, str], _Tracker] = {}  # by policy and role
        self._warned: set[str] = set()  # the strata columns a task lacked, warned of once
        self._warned_reward = False  # a reward that is no finite number, warned of once

    @staticmethod
    def check(options: RunningOptions, ctx: CreditCheckContext) -> None:
        """The entry's rows have a staleness bound (its ``max_staleness``, else ``train.max_staleness``): the
        statistics are kept as far back as the rows may be old, which needs one."""

        from platoon.train.entry_settings import staleness_bounds

        if staleness_bounds(ctx.config).get(ctx.entry) is None:
            raise ValueError(
                "it reads each agent's statistics as they were when the agent began sampling, kept as far back as the "
                "entry's rows may be old and still train, and they have no staleness bound; set train.max_staleness "
                "(how many policy versions old a rollout may be and still train)"
            )

    async def assign(self, group: CreditGroup) -> None:
        stratum = self._stratum(group)
        credited = []
        for agent in group.agents():
            reward = self._reward(agent, group)
            if reward is None:
                for sample in agent.samples:  # unscored: trains nothing
                    sample.set_advantage(0.0)
            elif agent.samples:
                credit = self._credit(group, agent, reward, stratum)
                for sample in agent.samples:
                    sample.set_advantage(credit.advantage)
                credited.append(credit)
        if credited:
            group.stats.scalar(**_metrics(credited))

    def settle(self, group: CreditGroup) -> None:
        """Settle each scored agent of the group into its role's statistics, at its policy's version now, with its
        residual against the baseline it was credited against (an agent without rows, or whose statistics are no
        longer kept: against the statistics now).  Every residual is taken before any of the group's samples settles,
        so that none moves another's."""

        stratum = self._stratum(group)
        settled = []
        for agent in group.agents():
            reward = self._reward(agent, group)
            if reward is None:
                continue
            tracker = self._tracker(agent)
            now = _version_now(group, agent.policy)
            read = tracker.at(_sampled_at(agent, group)) if agent.samples else None
            baseline = (read if read is not None else tracker.now(now)).baseline(stratum, self.options.shrink)
            settled.append((tracker, now, reward, reward - baseline))
        for tracker, now, reward, residual in settled:
            tracker.settle(now, stratum, reward, residual)

    def state(self) -> dict[str, Any]:
        return {
            "trackers": [
                [policy, role, tracker.to_json()] for (policy, role), tracker in sorted(self._trackers.items())
            ]
        }

    def load_state(self, state: dict[str, Any]) -> None:
        trackers = {}
        for policy, role, saved in state["trackers"]:
            policy, role = str(policy), str(role)
            trackers[(policy, role)] = _Tracker.from_json(
                saved, self._initial, self._window(policy), self.options.half_life
            )
        self._trackers = trackers

    def _tracker(self, agent: AgentRun) -> _Tracker:
        """The statistics of the agent's role of its policy, new ones (the priors) for its first."""

        key = (agent.policy, agent.role)
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = self._trackers[key] = _Tracker(self._initial, self._window(agent.policy), self.options.half_life)
        return tracker

    def _window(self, policy: PolicyId) -> int:
        """How many versions back a read of ``policy``'s statistics reaches: as many as its rows of the entry may be
        old and still train, and the newest (:data:`DEFAULT_WINDOW` when the setup gives no bound)."""

        bound = self.setup.max_staleness.get(policy)
        return DEFAULT_WINDOW if bound is None else bound + 1

    def _reward(self, agent: AgentRun, group: CreditGroup) -> float | None:
        """The agent's reward, or None when it has none (unscored) or one that is not a finite number, which a reward
        gone wrong upstream gives, and which would spoil every baseline it entered (warned of once)."""

        reward = agent.reward
        if reward is None or math.isfinite(reward):
            return reward
        if not self._warned_reward:
            self._warned_reward = True
            logger.warning(
                "environments[%s].credit: running: agent %s of task %s has a reward of %s, which is no finite number: "
                "it trains at 0 and enters no statistics, as an unscored agent (warned once)",
                group.entry,
                agent.agent_id,
                group.task_id,
                reward,
            )
        return None

    def _credit(self, group: CreditGroup, agent: AgentRun, reward: float, stratum: Stratum) -> _Credit:
        """The credit of an agent of reward ``reward`` (the module's docstring): read at the version it sampled at."""

        options = self.options
        now = group.versions.get(agent.policy, 0)
        sampled = _sampled_at(agent, group)
        tracker = self._trackers.get((agent.policy, agent.role))
        read = tracker.at(sampled) if tracker is not None else self._initial
        age = max(now - sampled, 0)
        if read is None:
            return _Credit(0.0, None, None, age, out_of_ring=True)
        baseline, scale = read.baseline(stratum, options.shrink), read.scale()
        if options.priors is None and read.count < WARMUP_SAMPLES:
            return _Credit(0.0, baseline, scale, age, warmup=True)
        advantage = (reward - baseline) / scale
        clipped = abs(advantage) > options.clip
        advantage = max(-options.clip, min(options.clip, advantage))
        capped = options.cap_penalties and reward < 0 and advantage > 0
        return _Credit(0.0 if capped else advantage, baseline, scale, age, clipped=clipped, capped=capped)

    def _stratum(self, group: CreditGroup) -> Stratum:
        """The group's task's stratum: its values of the strata columns, in order, down to the first it lacks."""

        values: list[str] = []
        for column in self.options.strata:
            value = group.columns.get(column)
            if value is None:
                if column not in self._warned:
                    self._warned.add(column)
                    logger.warning(
                        "environments[%s].credit_options.strata: task %s has no %s column, so the agents of a task "
                        "without it are credited at the stratum above it (warned once per column)",
                        group.entry,
                        group.task_id,
                        column,
                    )
                break
            values.append(value)
        return tuple(values)


def _sampled_at(agent: AgentRun, group: CreditGroup) -> int:
    """The version an agent began sampling at: the oldest of its rows' (one without a version is as new as its
    policy's version when ``assign`` began)."""

    now = group.versions.get(agent.policy, 0)
    return min((now if sample.version is None else sample.version for sample in agent.samples), default=now)


def _version_now(group: CreditGroup, policy: PolicyId) -> int:
    """A policy's version as a group of its settles: its version now, and at least its version when ``assign``
    began."""

    view = group.policies.get(policy)
    return max(view.version if view is not None else 0, group.versions.get(policy, 0))


def _metrics(credited: list[_Credit]) -> dict[str, float]:
    """The group's ``credit/*`` metrics, over the agents it credited (the module's docstring)."""

    count = len(credited)
    metrics = {
        "credit/running/advantage/avg": sum(credit.advantage for credit in credited) / count,
        "credit/agent_advantage/abs_avg": sum(abs(credit.advantage) for credit in credited) / count,
        "credit/running/age/avg": sum(credit.age for credit in credited) / count,
        "credit/running/warmup": sum(credit.warmup for credit in credited) / count,
        "credit/running/out_of_ring": sum(credit.out_of_ring for credit in credited) / count,
        "credit/running/clipped": sum(credit.clipped for credit in credited) / count,
        "credit/running/capped": sum(credit.capped for credit in credited) / count,
    }
    read = [credit for credit in credited if credit.baseline is not None and credit.scale is not None]
    if read:
        metrics["credit/running/baseline/avg"] = sum(credit.baseline or 0.0 for credit in read) / len(read)
        metrics["credit/running/scale/avg"] = sum(credit.scale or 0.0 for credit in read) / len(read)
    return metrics
