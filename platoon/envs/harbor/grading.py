"""How a Harbor verifier's output becomes an :class:`Outcome`: the dataset's reward contract and log classifiers.

A **reward contract** (``service_kwargs.reward``) names the reward that trains (``primary``) among the ones the
verifier reports, or lists the names that may train, for a dataset whose tasks report their score under different
names (one task's ``reward.txt`` is ``reward``, another's ``reward.json`` calls it ``correctness``).  With a list, a
reading must report exactly one listed name, from one reward file: two listed names (``verifier_ambiguous_primary``)
or both reward files (``verifier_ambiguous_reward_files``) make it invalid, so a policy whose code the tests run
cannot choose what trains by writing a second key or file, and the order of the list never matters.  The outcome
records the name that trained as its ``primary``.

A contract may also restrict which keys the verifier may report (``components``, which must include every primary
name) and the trained reward's range (``range``), and may name a ``check`` hook from the dataset's own package,
``fn(rewards, verifier_dir, task) -> Outcome | None``, that applies the dataset's rules (for example "correctness
counts only with a submission").  The hook gets the reported rewards, the local copy of ``/logs/verifier`` (a
directory named ``verifier`` that holds the reward files and the last MiB of each verifier log, the regular files
grading read: :mod:`platoon.envs.harbor.verifier`) and the :class:`~platoon.envs.harbor.task.HarborTask` (its
``index`` row and ``metadata``); it returns an outcome to use instead of platoon's reading, or None to keep it.  A
hook that raises is an ``error``, never a score.

**Log classifiers** (``service_kwargs.verifier_classifiers``) are opt-in multiline patterns over the verifier's
output (the last MiB of ``test-stdout.txt``, where platoon sends ``test.sh``'s stdout and stderr, and of
``test-stderr.txt``, ``stdout.txt`` and ``stderr.txt`` when the tests write them).  A match makes the outcome
``invalid`` with the classifier's ``kind``, whatever the reward file says: upstream graders often write 0 before
their tests run, so a grader that crashed reads like a policy that failed.  None is on by default: a test that runs
the agent's program under a timeout and lets the timeout escape measures a slow solution, not a broken grader.
platoon ships named sets for the common cases (:data:`CLASSIFIER_SETS`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from platoon.envs.harbor.verifier import LOG_TAIL_BYTES, REWARD_FILES, VERIFIER_LOGS, read_rewards
from platoon.envs.service import PLATOON_ZERO_KEY, Outcome
from platoon.registry import import_from_string

# The verifier's logs the classifiers read, in the local copy of /logs/verifier.
CLASSIFIED_LOGS = VERIFIER_LOGS
_LOG_TAIL_BYTES = LOG_TAIL_BYTES  # the classifiers read at most the last MiB of each log


@dataclass(frozen=True)
class LogClassifier:
    """A verifier-log pattern (a multiline regex) whose match makes the outcome ``invalid`` with ``kind``."""

    pattern: str
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError(f"A verifier classifier needs a pattern, got {self.pattern!r}")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"Verifier classifier {self.pattern!r} needs a kind (the failure kind it reports)")
        try:
            re.compile(self.pattern, re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"Verifier classifier {self.pattern!r} is not a valid regular expression: {exc}") from exc

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text, re.MULTILINE) is not None


# Named sets an environment can list in verifier_classifiers.  harbor-common holds the failures a downstream project
# found in upstream Harbor graders (in its reward contract): a grader subprocess that timed out, and a test runner
# missing from the image.
CLASSIFIER_SETS: dict[str, tuple[LogClassifier, ...]] = {
    "harbor-common": (
        LogClassifier(r"^subprocess\.TimeoutExpired:", "verifier_subprocess_timeout"),
        LogClassifier(
            r"^/tests/(?:test|run_tests)\.sh: line \d+: (?:uvx|uv|pytest): command not found\s*$",
            "verifier_missing_test_runner",
        ),
        LogClassifier(
            r"^(?:/[^\s:]+/)?python(?:3(?:\.\d+)?)?: No module named pytest\s*$", "verifier_missing_test_runner"
        ),
    ),
}

CheckHook = Callable[[dict[str, float], Path, Any], "Outcome | None"]


@dataclass(frozen=True)
class RewardContract:
    """Which reported reward trains and what makes a reading invalid (``service_kwargs.reward``)."""

    # The reward that trains (Outcome.primary), or a list of the names that may train: a reading must then report
    # exactly one of them (verifier_ambiguous_primary), from one reward file (verifier_ambiguous_reward_files).  A
    # scored outcome without one is an error (verifier_no_primary).
    primary: str | tuple[str, ...] = "reward"
    # The keys the verifier may report (None: any); another key makes the outcome invalid (verifier_unexpected_keys).
    components: tuple[str, ...] | None = None
    # Bounds of the reward that trained (None: any finite number); outside them the outcome is invalid
    # (verifier_out_of_range).
    range: tuple[float, float] | None = None
    # "module:function" of the dataset's check hook (see the module docstring), or None.
    check: str | None = None
    _hook: CheckHook | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.primary, list | tuple):
            names = tuple(self.primary)
            if not names or not all(isinstance(name, str) and name for name in names):
                raise ValueError(f"reward.primary must list reward names, got {self.primary!r}")
            if len(set(names)) < len(names):
                raise ValueError(f"reward.primary lists a reward twice: {list(names)}")
            object.__setattr__(self, "primary", names)
        elif not isinstance(self.primary, str) or not self.primary:
            raise ValueError(f"reward.primary must name a reward or list reward names, got {self.primary!r}")
        if self.components is not None:
            components = (self.components,) if isinstance(self.components, str) else tuple(self.components)
            if not all(isinstance(name, str) and name for name in components):
                raise ValueError(f"reward.components must list reward names, got {self.components!r}")
            if missing := [name for name in self.names if name not in components]:
                which = f"the primary {missing[0]!r}" if isinstance(self.primary, str) else f"every primary {missing}"
                raise ValueError(f"reward.components {list(components)} must include {which}")
            object.__setattr__(self, "components", components)
        if self.range is not None:
            bounds = tuple(self.range)
            if (
                len(bounds) != 2
                or not all(isinstance(b, int | float) and not isinstance(b, bool) and math.isfinite(b) for b in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"reward.range must be [low, high] with low <= high, got {self.range!r}")
            object.__setattr__(self, "range", (float(bounds[0]), float(bounds[1])))
        if self.check is not None:
            if not isinstance(self.check, str) or ":" not in self.check:
                raise ValueError(f"reward.check must be 'module:function', got {self.check!r}")
            try:
                hook = import_from_string(self.check)
            except (ImportError, AttributeError, ValueError) as exc:
                raise ValueError(f"reward.check {self.check!r} cannot be imported: {exc}") from exc
            if not callable(hook):
                raise ValueError(f"reward.check {self.check!r} is not callable")
            object.__setattr__(self, "_hook", hook)

    @property
    def names(self) -> tuple[str, ...]:
        """The names that may train: the primary alone, or the listed ones in their order."""

        return (self.primary,) if isinstance(self.primary, str) else self.primary

    def zero(self, reason: str, metadata: dict[str, Any] | None = None) -> Outcome:
        """A score of 0 that platoon gives itself, under the contract's first name: an attempt whose agents broke
        what grading needs (a reward file that is not a regular file, a workdir they made unusable).  Its metadata
        marks it (``PLATOON_ZERO_KEY``): no grader reported the name, so it trained no reward key."""

        return Outcome(
            status="scored",
            rewards={self.names[0]: 0.0},
            primary=self.names[0],
            reason=reason,
            metadata={**(metadata or {}), PLATOON_ZERO_KEY: True},
        )

    @classmethod
    def coerce(cls, value: RewardContract | Mapping[str, Any] | Sequence[str] | str | None) -> RewardContract:
        """A contract from config: a mapping of its fields, a primary key alone (``reward: correctness``), a list of
        primary names (``reward: [reward, correctness]``), or None."""

        if value is None:
            return cls()
        if isinstance(value, RewardContract):
            return value
        if isinstance(value, str):
            return cls(primary=value)
        if isinstance(value, list | tuple):
            return cls(primary=tuple(value))
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {"primary", "components", "range", "check"})
            if unknown:
                raise ValueError(f"reward has unknown keys {unknown}; it takes primary, components, range and check")
            return cls(**value)
        raise ValueError(f"reward must be a mapping, a reward name or a list of names, got {value!r}")

    def read(self, verifier_dir: Path, task: Any) -> Outcome:
        """The outcome of a verifier's reward files under this contract (see the module docstring)."""

        listed = not isinstance(self.primary, str)
        if listed and all((verifier_dir / name).is_file() for name in REWARD_FILES):
            return Outcome.infra(
                "verifier_ambiguous_reward_files",
                status="invalid",
                reason=f"the verifier left both {' and '.join(REWARD_FILES)}; with a list of primary rewards a reading "
                "comes from one reward file",
            )
        rewards = read_rewards(verifier_dir)
        if isinstance(rewards, Outcome):
            return rewards  # the files report no rewards: nothing for the contract to judge
        if self.components is not None and (unexpected := sorted(set(rewards) - set(self.components))):
            return Outcome.infra(
                "verifier_unexpected_keys",
                status="invalid",
                reason=f"the verifier reported {unexpected}, which the reward contract does not allow",
                metadata={"reported": rewards},
            )
        reported = [name for name in self.names if name in rewards]
        if not reported:
            wanted = (
                f"none of the primary rewards {list(self.names)}"
                if listed
                else f"not the primary reward {self.primary!r}"
            )
            return Outcome.infra(
                "verifier_no_primary",
                reason=f"the verifier reported {sorted(rewards)}, {wanted}",
                metadata={"reported": rewards},
            )
        if len(reported) > 1:
            return Outcome.infra(
                "verifier_ambiguous_primary",
                status="invalid",
                reason=f"the verifier reported {sorted(reported)}, more than one of the primary rewards "
                f"{list(self.names)}",
                metadata={"reported": rewards},
            )
        primary = reported[0]
        if self.range is not None and not self.range[0] <= rewards[primary] <= self.range[1]:
            return Outcome.infra(
                "verifier_out_of_range",
                status="invalid",
                reason=f"{primary} = {rewards[primary]:g} is outside {list(self.range)}",
                metadata={"reported": rewards},
            )
        scored = Outcome(status="scored", rewards=rewards, primary=primary)
        if self._hook is None:
            return scored
        try:
            checked = self._hook(dict(rewards), verifier_dir, task)
        except Exception as exc:
            return Outcome.infra(
                f"unexpected:{type(exc).__name__}",
                reason=f"reward check {self.check} raised: {exc}",
                metadata={"reported": rewards},
            )
        if checked is None:
            return scored
        if not isinstance(checked, Outcome):
            return Outcome.infra(
                "verifier_contract",
                reason=f"reward check {self.check} returned {type(checked).__name__}, not an Outcome or None",
                metadata={"reported": rewards},
            )
        return checked


def classifier_list(value: Any) -> tuple[LogClassifier, ...]:
    """Classifiers from config: a named set, a ``{pattern, kind}`` mapping, or a list of either (None: none)."""

    if value is None:
        return ()
    items: Sequence[Any] = [value] if isinstance(value, str | Mapping | LogClassifier) else value
    if not isinstance(items, Sequence):
        raise ValueError(f"verifier_classifiers must be a list, got {value!r}")
    classifiers: list[LogClassifier] = []
    for item in items:
        if isinstance(item, LogClassifier):
            classifiers.append(item)
        elif isinstance(item, str):
            if item not in CLASSIFIER_SETS:
                raise ValueError(f"Unknown verifier classifier set {item!r}; the sets are {sorted(CLASSIFIER_SETS)}")
            classifiers.extend(CLASSIFIER_SETS[item])
        elif isinstance(item, Mapping):
            unknown = sorted(set(item) - {"pattern", "kind"})
            if unknown:
                raise ValueError(f"A verifier classifier takes pattern and kind, not {unknown}")
            classifiers.append(LogClassifier(pattern=item.get("pattern"), kind=item.get("kind")))
        else:
            raise ValueError(f"verifier_classifiers entries are set names or {{pattern, kind}} tables, got {item!r}")
    return tuple(classifiers)


def classify(verifier_dir: Path, classifiers: Sequence[LogClassifier]) -> Outcome | None:
    """The ``invalid`` outcome of the first classifier that matches a verifier log, or None."""

    if not classifiers:
        return None
    for name in CLASSIFIED_LOGS:
        path = verifier_dir / name
        if not path.is_file():
            continue
        text = _tail(path)
        for classifier in classifiers:
            if classifier.matches(text):
                return Outcome.infra(
                    classifier.kind,
                    status="invalid",
                    reason=f"the verifier's {name} matches {classifier.pattern!r}",
                    metadata={"classifier": classifier.pattern},
                )
    return None


def _tail(path: Path) -> str:
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - _LOG_TAIL_BYTES))
        return file.read().decode("utf-8", errors="replace")
