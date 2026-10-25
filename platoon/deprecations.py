"""Deprecations: the warning platoon gives for a form that a later release removes, and the config keys going away.

A deprecated form keeps working for one minor release, with a :class:`PlatoonDeprecationWarning` that names what
replaces it and the release that removes it; the next minor release removes it, so a form that first warns in 0.3 is
gone in 0.4 (RFC 0016, section 10).

A config class names the keys it still takes but that are going away in a class attribute, ``DEPRECATED_KEYS``, as
platoon 0.3.0's ``TrainConfig`` named ``task_order``, the former name of ``task_sampler``::

    @dataclass
    class TrainConfig:
        DEPRECATED_KEYS: ClassVar[dict[str, DeprecatedKey]] = {
            "task_order": DeprecatedKey(removed_in="0.4.0", renamed_to="task_sampler", former_default="auto"),
        }

The config loader (:func:`platoon.utils.config.resolve_config`) finds every such key a config sets, wherever it sits,
and reports each as a :class:`Deprecation` with its dotted path: as a warning, in ``platoon config validate``'s report
(a failure with ``--strict``) and in the run's ``lock.json`` (``deprecations``).

The release named removes the key: the config class drops its field and names it in its ``UNKNOWN_KEY_HINTS``
(:mod:`platoon.utils.config`) with :func:`removed_hint`, so a config that still sets it does not load (exit 78), and
says what replaces it and the release that removed it, as platoon 0.4.0's ``TrainConfig`` names ``task_order``::

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "task_order": removed_hint("0.4.0", "train.task_sampler"),
    }

A service's removed options go in its class's ``UNKNOWN_KEY_HINTS`` alike: a config whose entry's ``service_kwargs``
give one does not load either (:func:`platoon.envs.service.service_option_hints`), and the check of an entry's options
against what its service takes says the same (:func:`platoon.envs.service.service_option_problems`).  A removed
command or variable fails with the same words.

A config that sets a renamed key and its new name to different values is a conflict that the config class refuses,
with one exception: the renamed key at the default it had before the rename (``former_default``), which a dump of an
older release's ``platoon config show --defaults`` spells out, yields to the new name.  The config class lets the new
name win then, and the deprecation says that the key's value was ignored.

The keys of a free-form mapping belong to whatever it configures, so a config class names how to find the deprecated
ones in a second class attribute, ``DEPRECATED_OPTIONS``: ``{field: function}``, the function taking the section that
holds the field and returning its ``{key: DeprecatedKey}``.  An environment entry's ``service_kwargs`` are found so,
from the ``DEPRECATED_KEYS`` of the service class the entry names.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "DEPRECATED_KEYS",
    "DEPRECATED_OPTIONS",
    "DeprecatedKey",
    "Deprecation",
    "PlatoonDeprecationWarning",
    "deprecated_keys",
    "removed_hint",
    "warn",
]

# The class attribute in which a config class names its deprecated keys: ``{key: DeprecatedKey(...)}``.
DEPRECATED_KEYS = "DEPRECATED_KEYS"
# The class attribute in which a config class names how to find the deprecated keys of its free-form mapping fields:
# ``{field: function}``, the function taking the section (a mapping) and returning ``{key: DeprecatedKey(...)}``.
DEPRECATED_OPTIONS = "DEPRECATED_OPTIONS"
# Frames in platoon's own package are skipped when a warning names where it came from.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class PlatoonDeprecationWarning(FutureWarning):
    """A form platoon still accepts, such as a config key, that a later release removes.

    Each warning names what replaces the form and the release that removes it. It is a ``FutureWarning``, which
    Python shows by default, unlike a ``DeprecationWarning`` outside ``__main__``. ``platoon config validate
    --strict`` fails on a config that uses a deprecated key, and in Python
    ``warnings.simplefilter("error", pt.PlatoonDeprecationWarning)`` (or pytest's
    ``-W error::platoon.PlatoonDeprecationWarning``) turns the warnings into errors.
    """


@dataclasses.dataclass(frozen=True)
class DeprecatedKey:
    """How a config key is going away: the release that removes it (``removed_in``, the minor release after the first
    one that warns about it), and what replaces it, either a key of the same section (``renamed_to``) or what to do
    instead, said in a few words (``instead``).  A renamed key may name the default it had before the rename
    (``former_default``): at that value it yields to its new name set to another value, and is ignored."""

    removed_in: str
    renamed_to: str | None = None
    instead: str | None = None
    former_default: Any = None

    def __post_init__(self) -> None:
        if (self.renamed_to is None) == (self.instead is None):
            raise ValueError("a deprecated key names what replaces it: give one of renamed_to and instead")
        if self.former_default is not None and self.renamed_to is None:
            raise ValueError("only a renamed key has a former default")

    def ignored_beside(self, value: Any, renamed_value: Any) -> bool:
        """Whether the key, set to ``value`` beside its new name set to ``renamed_value`` (None: not set), is ignored:
        it is at its former default, and the new name says otherwise."""

        return (
            self.former_default is not None
            and value == self.former_default
            and renamed_value is not None
            and renamed_value != value
        )

    def at(self, path: str, value: Any = None, renamed_value: Any = None) -> Deprecation:
        """The deprecation of a config that sets this key at the dotted ``path``, to ``value``, beside its new name set
        to ``renamed_value`` (None: not set), which says when the key's value is ignored (:meth:`ignored_beside`)."""

        if self.renamed_to is not None:
            parent = path.rpartition(".")[0]
            replacement = f"{parent}.{self.renamed_to}" if parent else self.renamed_to
        else:
            replacement = str(self.instead)
        ignored = (
            f"its value {value!r} is ignored, since {replacement} is {renamed_value!r}"
            if self.ignored_beside(value, renamed_value)
            else None
        )
        return Deprecation(path=path, replacement=replacement, removed_in=self.removed_in, ignored=ignored)


def deprecated_keys(schema: Any) -> Mapping[str, DeprecatedKey]:
    """The deprecated keys a config class names in its ``DEPRECATED_KEYS`` attribute; none for anything else."""

    found = getattr(schema, DEPRECATED_KEYS, None)
    return found if isinstance(found, Mapping) else {}


@dataclasses.dataclass(frozen=True)
class Deprecation:
    """A deprecated form that a config uses: the key's dotted ``path``, what replaces it and the release that
    removes it, and, when the config ignores the key's value (:meth:`DeprecatedKey.ignored_beside`), why
    (``ignored``)."""

    path: str
    replacement: str
    removed_in: str
    ignored: str | None = None

    @property
    def message(self) -> str:
        """One line, as a warning, ``platoon config validate`` and the command line give it."""

        message = f"{self.path}: deprecated; use {self.replacement} instead (platoon {self.removed_in} removes it)"
        return message if self.ignored is None else f"{message}; {self.ignored}"

    def to_dict(self) -> dict[str, str]:
        """The record ``lock.json`` keeps under ``deprecations``."""

        return {"path": self.path, "replacement": self.replacement, "removed_in": self.removed_in}


def warn(deprecations: Iterable[Deprecation]) -> None:
    """Warn of each deprecation, as a :class:`PlatoonDeprecationWarning` from the first caller outside platoon.

    Python's warning filters apply: an ``error`` filter raises the first one.
    """

    for deprecation in deprecations:
        warnings.warn(deprecation.message, PlatoonDeprecationWarning, skip_file_prefixes=(_PACKAGE_DIR,))


def removed_hint(removed_in: str, replacement: str) -> str:
    """What platoon says of a form that release ``removed_in`` removed, and what replaces it (``replacement``, as its
    deprecation named it): the hint of a removed key in a config class's ``UNKNOWN_KEY_HINTS``, quoted after its path
    and ``unknown key``, and the words of a removed command or variable."""

    return f"platoon {removed_in} removed it; use {replacement}"
