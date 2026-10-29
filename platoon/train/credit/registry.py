"""Which credit estimator each environment entry credits its groups with: the ``credit`` component kind (RFC 0013's one
discovery rule), and what a config chooses (``environments[].credit`` with its ``credit_options``, RFC 0016).

An estimator is named by core's name (:data:`BUILTIN_CREDIT_ESTIMATORS`: ``group``, ``keyed``, ``reverse_kl``), by a
name that a module an entry's ``package`` names registers (:func:`credit_estimator`), or as ``module:Class``
(:func:`resolve_credit_estimator`).  When the config loads, each entry's estimator resolves, its options are parsed
against its schema and its ``check`` runs (:func:`choose_credit`, exit 78 for any problem), and ``lock.json`` records
it (:func:`credit_lock_records`); ``platoon train`` then builds each entry's estimator for the training workflow
(:func:`build_entry_credit`), with the run's policies as estimators read them (:func:`policy_views`).  An entry without
``credit`` keeps the workflow's own credit, today's ``group``, run as it always ran; in a run that trains several
policies, ``keyed`` on ``[task, policy, role]`` instead (RFC 0021's D5: each agent of each policy and role on its own
baseline), which ``lock.json`` records as it records a chosen one.  Core's built-ins are imported by name, so this
module imports none of them.
"""

from __future__ import annotations

import difflib
import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from platoon.train.credit.api import CreditCheckContext, CreditSetup, PolicyView
from platoon.train.policy.spec import MAIN, PolicyId, policy_ids, trained_policies

R = TypeVar("R", bound=type)

# The registry kind of credit estimators (platoon.registry), which @credit_estimator registers in.
REGISTRY_KIND = "credit"
# Core's estimators, by name: resolved from here before the registry is read, so that nothing registered can shadow
# them, whatever the import order.
BUILTIN_CREDIT_ESTIMATORS: dict[str, str] = {
    "group": "platoon.train.credit.group:GroupCredit",
    "keyed": "platoon.train.credit.keyed:Keyed",
    "reverse_kl": "platoon.train.credit.reverse_kl:ReverseKL",
}
# Core's names, that of the estimator platoon adds included (gae): no package may register them.
RESERVED_CREDIT_NAMES = frozenset({*BUILTIN_CREDIT_ESTIMATORS, "gae"})
# Core's estimators, as messages name them.
_CORE = "group, keyed and reverse_kl"
# The estimator of an entry that names none, in a run that trains several policies (RFC 0021, D5).
SEVERAL_POLICIES_DEFAULT = "keyed"
# What a registered name may be: no spaces, and neither ``:`` (module:Class) nor ``@``.
_NAME = re.compile(r"[^\s:@]+")


def credit_estimator(
    name: str, *, options: Any = None, check: Callable[[Any, CreditCheckContext], None] | None = None
) -> Callable[[R], R]:
    """Register a credit estimator class under ``name``, from a module that an environment entry's ``package`` names,
    so that configs can name it (``credit: <name>``); ``options`` sets its ``options_schema`` and ``check`` its
    load-time check.  Core's names (``group``, ``keyed``, ``reverse_kl``, and ``gae``, which platoon adds) are
    reserved, whatever their case and their ``-`` or ``_``; a name registered twice is an error."""

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a credit estimator's name is a word without spaces, ':' or '@', got {name!r}")
    if _plain(name) in RESERVED_CREDIT_NAMES:
        raise ValueError(
            f"credit estimator name {name!r} is core's ({', '.join(sorted(RESERVED_CREDIT_NAMES))} are reserved): "
            "register yours under another name"
        )

    def decorator(cls: R) -> R:
        from platoon.registry import register_component

        if not isinstance(cls, type):
            raise TypeError(f"@credit_estimator({name!r}) registers a class, got {cls!r}")
        if options is not None:
            setattr(cls, "options_schema", options)
        if check is not None:
            setattr(cls, "check", staticmethod(check))
        register_component(REGISTRY_KIND, name, cls)
        return cls

    return decorator


def _plain(name: str) -> str:
    """A name as it reads, whatever its case and its ``-`` or ``_``: ``Reverse-KL`` reads as ``reverse_kl``."""

    return name.casefold().replace("-", "_")


def resolve_credit_estimator(ref: str) -> type:
    """The estimator class ``ref`` names: core's (:data:`BUILTIN_CREDIT_ESTIMATORS`), then a name a package registered
    (:func:`credit_estimator`), then ``module:Class``.  An :class:`~platoon.extensions.ExtensionError` that says why for
    anything else (with the nearest name), and for a class that is no estimator (:func:`estimator_problem`)."""

    from platoon.extensions import ExtensionError
    from platoon.registry import get_registry, import_from_string

    if not isinstance(ref, str) or not ref:
        raise ExtensionError(f"a credit estimator is named by a non-empty string, got {ref!r}")
    if ref in BUILTIN_CREDIT_ESTIMATORS:
        return import_from_string(BUILTIN_CREDIT_ESTIMATORS[ref])
    if _plain(ref) in BUILTIN_CREDIT_ESTIMATORS:
        raise ExtensionError(f"unknown credit estimator {ref!r}; did you mean {_plain(ref)}? Core's are {_CORE}")
    if _plain(ref) in RESERVED_CREDIT_NAMES:
        raise ExtensionError(f"credit estimator {ref!r} is reserved for one platoon will add; core's are {_CORE}")
    registry = get_registry(REGISTRY_KIND)
    if ":" not in ref:
        if ref not in registry.names():
            close = difflib.get_close_matches(ref, [*BUILTIN_CREDIT_ESTIMATORS, *registry.names()], n=1, cutoff=0.6)
            raise ExtensionError(
                f"unknown credit estimator {ref!r}"
                + (f"; did you mean {close[0]}?" if close else ";")
                + f" Core's: {_CORE}; a package an entry names registers others with @credit_estimator, or "
                "name one as module:Class"
            )
        cls = registry.get(ref)
    else:
        try:
            cls = import_from_string(ref)
        except Exception as exc:
            raise ExtensionError(f"{ref} does not import ({type(exc).__name__}: {exc})") from exc
    if (problem := estimator_problem(cls)) is not None:
        raise ExtensionError(f"{ref} is not a credit estimator: {problem}")
    return cls


def estimator_problem(cls: Any) -> str | None:
    """Why ``cls`` is no credit estimator class, or None when it is one: a class whose ``assign(group)`` is a coroutine
    function and whose ``needs_agents`` is a bool, whose ``settle``, if any, is synchronous, and that has both or
    neither of ``state()`` and ``load_state(state)``."""

    if not isinstance(cls, type) or not inspect.iscoroutinefunction(getattr(cls, "assign", None)):
        return "a class with async assign(group)"
    if not isinstance(getattr(cls, "needs_agents", None), bool):
        return "its needs_agents is no bool (True when it reads the group's agents)"
    settle = getattr(cls, "settle", None)
    if settle is not None and (not callable(settle) or inspect.iscoroutinefunction(settle)):
        return "its settle(group) is no synchronous method: the run settles a group with no await"
    if callable(getattr(cls, "state", None)) != callable(getattr(cls, "load_state", None)):
        return "it has one of state() and load_state(state) without the other: a state to resume needs both"
    return None


# --- what a config chooses --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditChoice:
    """What an entry chooses, resolved: the estimator's name as the config gives it, its class, its options parsed
    (None: it takes none) and as the config wrote them; ``default``: the config names none, and the run trains several
    policies (:data:`SEVERAL_POLICIES_DEFAULT`)."""

    entry: str
    name: str
    cls: type
    options: Any = None
    written: Mapping[str, Any] = field(default_factory=dict)
    default: bool = False

    def options_json(self) -> dict[str, Any]:
        """Its options as ``lock.json`` records them: parsed, defaults included."""

        from platoon.options import schema_options

        return schema_options(_schema(self.cls), name=self.name).as_json(self.options, self.written)


def choose_credit(config: Any, *, check: bool = True) -> dict[str, CreditChoice]:
    """The estimators a config's entries choose (``environments[].credit``), by entry label in config order, each
    resolved (:func:`resolve_credit_estimator`), its options (``credit_options``) parsed against its schema and its
    ``check`` run (unless ``check`` is False: ``platoon train`` building them after the config was checked); none for
    a config whose entries set none, unless it trains several policies: then an entry that sets none credits with
    ``keyed`` and its default options (:data:`SEVERAL_POLICIES_DEFAULT`).  A
    :class:`~platoon.utils.config.ConfigError` with every problem, each with its path.  The entries' packages must be
    imported first, as a run imports them."""

    from platoon.extensions import ExtensionError
    from platoon.options import schema_options
    from platoon.utils.config import ConfigError

    problems: list[str] = []
    choices: dict[str, CreditChoice] = {}
    several = len(trained_policies(config)) > 1
    for index, environment in enumerate(getattr(config, "environments", None) or ()):
        name = getattr(environment, "credit", None)
        if name is None and not several:
            continue
        label, where, default = environment.label, f"environments.{index}", name is None
        if name is None:  # each row trains its own policy: credit tells which, per agent
            name = SEVERAL_POLICIES_DEFAULT
        try:
            cls = resolve_credit_estimator(name)
        except ExtensionError as exc:
            problems.append(f"{where}.credit: {exc}")
            continue
        written = dict(environment.credit_options)
        parsed, found = schema_options(_schema(cls), name=name).check(written, f"{where}.credit_options")
        problems += found
        method = getattr(cls, "check", None)
        if check and not found and callable(method):
            try:
                method(parsed, CreditCheckContext(entry=label, policies=policy_ids(config), config=config))
            except Exception as exc:
                problems.append(f"{where}.credit: {name}: {str(exc) or type(exc).__name__}")
        choices[label] = CreditChoice(label, name, cls, parsed, written, default)
    if problems:
        raise ConfigError(problems)
    return choices


def _schema(cls: type) -> Any:
    return getattr(cls, "options_schema", None)


def workflow_problems(choices: Mapping[str, CreditChoice], workflow_cls: Any, reference: str) -> list[str]:
    """An entry's estimator beside a training workflow class that does not apply it, which would credit the entry's
    groups its own way and ignore the estimator without a word: one that does not declare ``applies_credit = True``
    (its capabilities, :func:`~platoon.train.workflow.protocol.capabilities_of`), such as a ``GroupRolloutWorkflow``
    subclass that overrides ``compute_baselines`` or ``assign_advantages``."""

    from platoon.train.workflow.protocol import capabilities_of

    if not choices or capabilities_of(workflow_cls).applies_credit:
        return []
    why = (
        "(it does not declare applies_credit = True; a GroupRolloutWorkflow subclass that overrides compute_baselines "
        "or assign_advantages credits with those)"
    )
    return [
        f"environments[{label}].credit: the run trains several policies, so its groups are credited with an estimator "
        f"that tells each row's policy ({choice.name}, by default), and the run's workflow, {reference}, applies none "
        f"{why}; use a workflow that applies it"
        if getattr(choice, "default", False)
        else f"environments[{label}].credit: the run's workflow, {reference}, does not apply an entry's credit "
        f"estimator {why}, so it would be ignored; leave credit out, or use a workflow that applies it"
        for label, choice in choices.items()
    ]


def policy_views(policies: Mapping[PolicyId, Any]) -> dict[PolicyId, PolicyView]:
    """Each of the run's policies (its ``PolicySet``) as an estimator may read it, by id (:func:`_view`)."""

    return {policy_id: _view(policy) for policy_id, policy in policies.items()}


def _view(policy: Any) -> PolicyView:
    """A policy of the run (a ``Policy``) as an estimator may read it: its id, its renderer and tokenizer, its version
    as it moves on (``Policy.version``), and its scores of other rows (``Policy.score``)."""

    llm = policy.info.llm
    return PolicyView(
        policy.id,
        renderer=getattr(llm, "renderer", None),
        tokenizer=getattr(llm, "tokenizer", None),
        scorer=policy.score,
        current_version=lambda: policy.version,
    )


def build_entry_credit(
    choices: Mapping[str, CreditChoice],
    settings: Mapping[str, Any],
    policy_views: Mapping[PolicyId, PolicyView] | None = None,
) -> dict[str, Any]:
    """Each entry's estimator, by entry label, built as ``Class(setup, options)`` with the entry's training settings
    (``settings``: each entry's :class:`~platoon.train.entry_settings.EntrySettings`, by label), and the run's policies
    with their views (``policy_views``, the trainer's; without them, the one policy ``main``)."""

    views = dict(policy_views or {})
    policies = tuple(views) or (MAIN,)
    return {
        label: choice.cls(
            CreditSetup(
                name=choice.name, entry=label, settings=settings[label].train, policies=policies, policy_views=views
            ),
            choice.options,
        )
        for label, choice in choices.items()
    }


def credit_lock_records(choices: Mapping[str, CreditChoice]) -> list[dict[str, Any]]:
    """What ``lock.json`` records of the entries' estimators (``kind: credit``), one record per estimator: the
    reference as the config gives it, what it resolved to, the distribution and version that provide it, the entries
    that use it and each one's options, defaults included.  None in a run whose entries set no ``credit``."""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for label, choice in choices.items():
        source = f"{choice.cls.__module__}:{choice.cls.__qualname__}"
        if (choice.name, source) not in records:
            records[(choice.name, source)] = _lock_record(choice.name, source, choice.cls.__module__)
        records[(choice.name, source)]["environments"].append(label)
        records[(choice.name, source)]["options"][label] = choice.options_json()
    return list(records.values())


def _lock_record(ref: str, source: str, module: str) -> dict[str, Any]:
    """The ``lock.json`` record of an estimator before its entries: the reference, what it resolved to, and the
    distribution and version that provide its ``module``."""

    from platoon.extensions import origin_of_module

    origin = origin_of_module(module)
    return {
        "kind": REGISTRY_KIND,
        "ref": ref,
        "resolved": source,
        "source": source,
        "distribution": origin.distribution,
        "distribution_version": origin.version,
        "entry_point": None,
        "environments": [],
        "options": {},
    }
