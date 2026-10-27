"""A policy as the config declares it (:class:`PolicySpec`), the run's policies as its config declares them
(:func:`policy_specs`), which policy each agent samples (:class:`AgentBindings`, :func:`agent_bindings`) and which
tools each agent role may call (:func:`role_tools`), and what loading refuses of them (:func:`policy_problems`, and
:func:`workflow_problems` against the run's workflow class) or warns of (:func:`policy_warnings`).  Imports no Tinker.

``main`` is the policy ``train:`` declares: each field of its spec keeps its ``train.*`` name, so that one vocabulary
names a policy wherever it is declared.  ``policies:`` declares the run's others under the same names (RFC 0021).  This
release trains ``main`` only: every other policy is frozen, sampled by the agents ``agents:`` binds to it or scoring
for a credit estimator, and never stepped, saved or pruned.
"""

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from platoon.train import task_stats
from platoon.utils.judges import JUDGE_ROLE, VERIFIER_ROLE
from platoon.utils.role_tools import unmatched

if TYPE_CHECKING:
    from platoon.train.config_defs import AdamParams

PolicyId = str
# The policy that ``train:`` declares.  The same object as ``task_stats.MAIN``, by whose versions the per-task
# statistics age their evidence, so that their snapshots keep their format.
MAIN: Final[PolicyId] = task_stats.MAIN


@dataclass(frozen=True)
class PolicySpec:
    """One policy, as the config declares it: the model, how its prompts are rendered and sampled, its Tinker service,
    how a new run trains it (its adapter's rank, the weights it starts from), and how each step updates it (its
    optimizer, and the loss its rows train under)."""

    id: PolicyId
    model_name: str
    renderer_name: str
    # The config's, as the renderer takes them.  A dict does not hash, so the spec's hash leaves it out.
    renderer_kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    context_window_length: int | None = None
    sample_timeout_seconds: float | None = None
    # Its Tinker service; None: TINKER_BASE_URL, else hosted Tinker.
    tinker_base_url: str | None = None
    # Whether the run trains it.  A frozen policy is sampled or scores, and is never stepped, saved or pruned.
    trainable: bool = True
    lora_rank: int = 32
    # The weights a new run starts from; None: the base model's.  A frozen policy's: the sampler it serves.
    load_checkpoint_path: str | None = None
    # With load_checkpoint_path: whether the new run takes that checkpoint's optimizer state too (Adam's moments), a
    # new trial that goes on from it; else its optimizer starts fresh.
    load_optimizer_state: bool = False
    # Its optimizer's settings, the loss its rows train under (its objective's, platoon.train.objectives) and the
    # loss's keyword arguments, for a policy the run trains (main's are train.optimizer, train.loss_fn and
    # train.loss_fn_config, which policy_specs puts here, so that the trainer reads them from the spec alone).
    optimizer: "AdamParams | None" = field(default=None, hash=False)
    loss_fn: str | None = None
    loss_fn_config: dict[str, Any] | None = field(default=None, hash=False)


def policy_specs(config: Any) -> dict[PolicyId, PolicySpec]:
    """The run's policies, in order: ``main``, as ``train:`` declares it (``train.model_name``, ``renderer_name``,
    ``renderer_kwargs``, ``context_window_length``, ``sample_timeout_seconds``, ``lora_rank``, ``optimizer``,
    ``loss_fn`` and ``loss_fn_config``), with the run's ``tinker_base_url``, and the weights
    ``checkpoint.load_checkpoint_path`` starts it from, with their optimizer state when
    ``checkpoint.load_optimizer_state`` says so; then each ``policies:`` entry, in the config's order, a frozen one in
    this release, whose unset service is the run's (``tinker_base_url``)."""

    train = config.train
    main = PolicySpec(
        id=MAIN,
        model_name=train.model_name,
        renderer_name=train.renderer_name,
        renderer_kwargs=train.renderer_kwargs,
        context_window_length=train.context_window_length,
        sample_timeout_seconds=train.sample_timeout_seconds,
        tinker_base_url=config.tinker_base_url,
        lora_rank=train.lora_rank,
        load_checkpoint_path=config.checkpoint.load_checkpoint_path,
        load_optimizer_state=config.checkpoint.load_optimizer_state,
        optimizer=train.optimizer,
        loss_fn=train.loss_fn,
        loss_fn_config=train.loss_fn_config,
    )
    specs = {MAIN: main}
    for policy_id, declared in _declared(config).items():
        specs[policy_id] = PolicySpec(
            id=policy_id,
            model_name=declared.model_name,
            renderer_name=declared.renderer_name,
            renderer_kwargs=declared.renderer_kwargs,
            context_window_length=declared.context_window_length,
            sample_timeout_seconds=declared.sample_timeout_seconds,
            tinker_base_url=_or(declared.tinker_base_url, config.tinker_base_url),
            trainable=declared.trainable,
            load_checkpoint_path=declared.load_checkpoint_path,
        )
    return specs


def policy_ids(config: Any) -> tuple[PolicyId, ...]:
    """The ids of the run's policies, in order (:func:`policy_specs`'s, without building the specs)."""

    return (MAIN, *_declared(config))


def _or(value: Any, default: Any) -> Any:
    return default if value is None else value


def _declared(config: Any) -> Mapping[PolicyId, Any]:
    """The config's ``policies:`` entries, by id (none for a config built without the key)."""

    return getattr(config, "policies", None) or {}


@dataclass(frozen=True)
class AgentBindings:
    """Which policy each agent samples, for its whole life (``agents:``): the root agent samples ``default``; an agent
    opened with a role samples that role's policy (``roles``), else ``default``.  A request never picks a policy."""

    default: PolicyId = MAIN
    roles: Mapping[str, PolicyId] = field(default_factory=dict, hash=False)

    def sampled(self) -> tuple[PolicyId, ...]:
        """The policies some agent samples: ``default`` first, then the roles' others, in their order."""

        return tuple(dict.fromkeys((self.default, *self.roles.values())))


def agent_bindings(config: Any) -> AgentBindings:
    """Which policy each agent of the run samples, as ``agents:`` says (without it, every agent samples ``main``)."""

    agents = getattr(config, "agents", None)
    return AgentBindings(
        default=getattr(agents, "policy", None) or MAIN,
        roles={role: bound.policy for role, bound in _roles(config).items() if bound.policy is not None},
    )


def role_tools(config: Any) -> dict[str, tuple[str, ...]]:
    """The tools the agents of each role may see and call, as ``agents.roles.<role>.tools`` lists them (names or
    ``fnmatch`` patterns, :mod:`platoon.utils.role_tools`), for the roles that list them: none without (every agent
    may call every tool)."""

    return {role: tuple(bound.tools) for role, bound in _roles(config).items() if bound.tools is not None}


def _roles(config: Any) -> Mapping[str, Any]:
    """The config's ``agents.roles`` entries, by role (none for a config built without the key)."""

    return getattr(getattr(config, "agents", None), "roles", None) or {}


# A policy's id: a lowercase letter, then lowercase letters, digits, ``_`` and ``-``, 32 characters at most.
POLICY_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}")
# The root agent's role (the gateway's AgentSession.role), whose agent samples agents.policy whatever agents.roles says.
ROOT_ROLE = "policy"
# The roles platoon's own code opens agents with: the root agent's, that of a child a harness spawns unless it names
# another (ctx.agents.spawn), a verifier's and a judge's.  A harness may mint agents of any other role.
PLATOON_ROLES: Final = (ROOT_ROLE, "subagent", VERIFIER_ROLE, JUDGE_ROLE)


def policy_problems(config: Any) -> list[str]:
    """What loading refuses of the config's ``policies:`` and ``agents:``, one entry each: an id that is no policy id
    (:data:`POLICY_ID`) or that names ``main``, which ``train:`` declares; a second policy the run would train, and
    a default policy of the agents other than main, which this release does not support (main publishes its binding
    as the agents' default); an agent role, or the agents' default, bound to a policy the run does not declare; and a
    role's ``tools`` that is no list of names or patterns."""

    problems = []
    declared = _declared(config)
    for policy_id, policy in declared.items():
        where = f"policies.{policy_id}"
        if policy_id == MAIN:
            problems.append(
                f"{where}: main is the policy train: declares (train.model_name and the other train.* model keys); "
                "give this one another id"
            )
        elif not isinstance(policy_id, str) or not POLICY_ID.fullmatch(policy_id):
            problems.append(
                f"{where}: a policy id is a lowercase letter, then lowercase letters, digits, '_' or '-', 32 "
                "characters at most"
            )
        elif getattr(policy, "trainable", True):
            problems.append(
                f"{where}: this release trains one policy, main (train:), so every other policy is frozen: set "
                f"{where}.trainable: false to sample it or score with it. Training several policies arrives with "
                "lockstep multi-policy training (RFC 0021, milestone B4)"
            )
    known = {MAIN, *declared}
    agents = getattr(config, "agents", None)
    default = getattr(agents, "policy", None)
    if default is not None and default not in known:
        problems.append(f"agents.policy: {default!r} is not a policy of the run ({', '.join(sorted(known))})")
    elif default not in (None, MAIN):
        problems.append(
            f"agents.policy: {default!r} is frozen, and the root agent samples the policy the run trains, main, in "
            "this release; bind a role to it (agents.roles.<role>.policy) instead. A default policy of another id "
            "arrives with lockstep multi-policy training (RFC 0021, milestone B4)"
        )
    for role, bound in _roles(config).items():
        if not isinstance(role, str) or not role:
            problems.append(f"agents.roles: a role is a non-empty name, got {role!r}")
            continue
        if bound.policy is not None and bound.policy not in known:
            problems.append(
                f"agents.roles.{role}.policy: {bound.policy!r} is not a policy of the run ({', '.join(sorted(known))})"
            )
        problems.extend(_tools_problems(role, bound.tools))
    return problems


def _tools_problems(role: str, tools: Any) -> list[str]:
    """What is wrong with ``agents.roles.<role>.tools``: a value that is no list, or an item that is no non-empty
    string (a name or a pattern)."""

    where = f"agents.roles.{role}.tools"
    if tools is None:
        return []
    if not isinstance(tools, list | tuple):
        return [f"{where}: a list of tool names or patterns (read_*), got {tools!r}; [] allows no tool"]
    return [
        f"{where}: a tool name or pattern is a non-empty string, got {item!r}"
        for item in tools
        if not isinstance(item, str) or not item
    ]


def workflow_problems(config: Any, workflow_cls: Any, reference: str) -> list[str]:
    """Agents bound to several policies beside a workflow class that cannot bind them: one that does not declare
    ``binds_policies`` (its capabilities, :func:`~platoon.train.workflow.protocol.capabilities_of`), whose rollouts
    would sample one policy only."""

    from platoon.train.workflow.protocol import capabilities_of  # the workflow contract, read at load only

    sampled = agent_bindings(config).sampled()
    if len(sampled) < 2 or capabilities_of(workflow_cls).binds_policies:
        return []
    return [
        f"agents: the agents sample several policies ({', '.join(sampled)}), and the run's workflow, {reference}, "
        "binds one (it does not declare binds_policies = True): leave agents: out, or use a workflow that binds them"
    ]


def policy_warnings(config: Any) -> list[str]:
    """What loading warns of the config's ``policies:`` and ``agents:``, one line each: a frozen policy that no agent
    samples and no entry's ``credit_options`` names (a teacher's), which the run opens and never uses; a policy
    bound to the role ``policy``, the root agent's, which the root never samples (it samples ``agents.policy``); a
    role that looks like a misspelling of one of :data:`PLATOON_ROLES` (:func:`_role_typos`); and a role's tool
    pattern that matches no tool of the run's environments, when each declares its tools up front
    (:func:`_tool_warnings`)."""

    agents = agent_bindings(config)
    used = set(agents.sampled())
    for environment in getattr(config, "environments", None) or ():
        used |= {value for value in (environment.credit_options or {}).values() if isinstance(value, str)}
    lines = [
        f"policies.{policy}: no agent samples it (agents.roles) and no entry's credit_options names it, so the run "
        "opens it and never uses it"
        for policy in _declared(config)
        if policy not in used
    ]
    if agents.roles.get(ROOT_ROLE, agents.default) != agents.default:
        lines.append(
            f"agents.roles.{ROOT_ROLE}.policy: the root agent's role is {ROOT_ROLE}, and the root samples "
            f"agents.policy ({agents.default}) whatever its role: only a subagent minted with the role {ROOT_ROLE} "
            "samples this policy"
        )
    return [*lines, *_role_typos(config), *_tool_warnings(config)]


def _role_typos(config: Any) -> list[str]:
    """A role of ``agents.roles`` that is none of :data:`PLATOON_ROLES` but close to one (``subagents``): its settings
    reach only the agents a harness mints with that very name, so it is likely misspelled."""

    lines = []
    for role in _roles(config):
        if not isinstance(role, str) or role in PLATOON_ROLES:
            continue
        if close := difflib.get_close_matches(role.lower(), PLATOON_ROLES, n=1, cutoff=0.8):
            lines.append(
                f"agents.roles.{role}: platoon opens no agent of this role, so its settings reach only the agents a "
                f"harness mints with it; did you mean {close[0]!r}?"
            )
    return lines


def _tool_warnings(config: Any) -> list[str]:
    """A pattern of ``agents.roles.<role>.tools`` that matches none of the tools the run's environments serve, when
    every environment declares its tools up front (:func:`~platoon.envs.service.service_declared_tools`): that pattern
    allows nothing."""

    tools = role_tools(config)
    declared = _declared_tools(config) if tools else None
    if declared is None:
        return []
    names = ", ".join(sorted(declared)) or "none"
    return [
        f"agents.roles.{role}.tools: {pattern!r} matches no tool of the run's environments ({names}), so it allows "
        "nothing"
        for role, patterns in tools.items()
        for pattern in unmatched(patterns, declared)
    ]


def _declared_tools(config: Any) -> set[str] | None:
    """The names of the tools the run's environments serve, when each declares them up front; else None."""

    from platoon.envs.service import service_declared_tools  # imports the environments' classes: at load only

    names: set[str] = set()
    for environment in getattr(config, "environments", None) or ():
        service = getattr(environment, "service", None)
        declared = service_declared_tools(service) if isinstance(service, str) else None
        if declared is None:
            return None
        names |= declared
    return names
