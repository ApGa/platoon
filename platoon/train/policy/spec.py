"""A policy as the config declares it (:class:`PolicySpec`), the run's policies as its config declares them
(:func:`policy_specs`), which policy each agent samples (:class:`AgentBindings`, :func:`agent_bindings`) and which
tools each agent role may call (:func:`role_tools`), the bound the dispatcher's horizon counts (:func:`horizon_bound`),
how the trained policies step (:data:`Stepping`, :func:`steps_independently`), and what loading refuses of them
(:func:`policy_problems`, and :func:`workflow_problems` against the run's workflow class) or warns of
(:func:`policy_warnings`).  Imports no Tinker.

``main`` is the policy ``train:`` declares: each field of its spec keeps its ``train.*`` name, so that one vocabulary
names a policy wherever it is declared.  ``policies:`` declares the run's others under the same names (RFC 0021): ones
the run trains with main, each with its own optimizer, loss and staleness bound (and, stepping independently, its own
batch and checkpoint cadence), and frozen ones, sampled by the agents ``agents:`` binds to them or scoring for a credit
estimator, and never stepped, saved or pruned.
"""

import difflib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from platoon.train import task_stats
from platoon.utils.judges import JUDGE_ROLE, VERIFIER_ROLE
from platoon.utils.role_tools import unmatched

if TYPE_CHECKING:
    from platoon.train.config_defs import AdamParams, PolicyCheckpointConfig

PolicyId = str
# How a run's trained policies step (train.stepping, a trainer's TrainerNeeds.stepping): all together, in each step
# (lockstep), or each on its own clock (independent).
Stepping = Literal["lockstep", "independent"]
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
    # loss's keyword arguments, for a policy the run trains (each unset one train.optimizer, train.loss_fn and
    # train.loss_fn_config, which policy_specs puts here, so that the trainer reads them from the spec alone).
    optimizer: "AdamParams | None" = field(default=None, hash=False)
    loss_fn: str | None = None
    loss_fn_config: dict[str, Any] | None = field(default=None, hash=False)
    # How many of its own versions old its rows may be and still train; None: their entry's bound, else
    # train.max_staleness (the stale filter's, platoon.train.step.staleness).
    max_staleness: int | None = None
    # Stepping independently (train.stepping: independent), how many of its shares (its rows of a group) each of its
    # steps trains; None: train.batch_size, the leader's.
    batch_size: int | None = None
    # Stepping independently, when it saves full checkpoints, counted in its own steps; None: the run's checkpoint.every
    # and checkpoint.recovery_every, in its steps.
    checkpoint: "PolicyCheckpointConfig | None" = field(default=None, hash=False)


def service_name(spec: PolicySpec) -> str:
    """The policy's Tinker service, as a message names it: its URL, else the one its client finds, ``TINKER_BASE_URL``'s
    when that is set, else hosted Tinker."""

    if spec.tinker_base_url:
        return spec.tinker_base_url
    found = os.environ.get("TINKER_BASE_URL")
    return f"{found} (TINKER_BASE_URL)" if found else "hosted Tinker"


def failure_note(spec: PolicySpec, doing: str) -> str:
    """The note an error of a policy's service gets in a run of several trained policies: whose service failed, and
    at what (``doing``)."""

    return f"policy {spec.id}, on {service_name(spec)}: {doing} failed"


def policy_specs(config: Any) -> dict[PolicyId, PolicySpec]:
    """The run's policies, in order: ``main``, as ``train:`` declares it (``train.model_name``, ``renderer_name``,
    ``renderer_kwargs``, ``context_window_length``, ``sample_timeout_seconds``, ``lora_rank``, ``optimizer``,
    ``loss_fn`` and ``loss_fn_config``), with the run's ``tinker_base_url``, and the weights
    ``checkpoint.load_checkpoint_path`` starts it from, with their optimizer state when
    ``checkpoint.load_optimizer_state`` says so; then each ``policies:`` entry, in the config's order, whose unset
    service is the run's (``tinker_base_url``), and, when the run trains it, whose unset adapter rank, optimizer and
    loss are ``train:``'s (:func:`_trained_settings`)."""

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
            **(_trained_settings(declared, train) if declared.trainable else {}),
        )
    return specs


def _trained_settings(declared: Any, train: Any) -> dict[str, Any]:
    """A trained ``policies:`` entry's training settings, as its spec holds them: its adapter's rank, else
    ``train.lora_rank``; its optimizer, else ``train.optimizer``; its loss, else ``train.loss_fn``; its loss's config,
    else ``train.loss_fn_config`` when the loss is ``train:``'s, and none (the loss's own defaults) for another loss,
    whose keys ``train:``'s config is not for; its staleness bound; and whether it starts from its checkpoint's
    optimizer state."""

    loss_fn = _or(declared.loss_fn, train.loss_fn)
    loss_fn_config = declared.loss_fn_config
    if loss_fn_config is None:
        loss_fn_config = train.loss_fn_config if loss_fn == train.loss_fn else {}
    return {
        "lora_rank": _or(declared.lora_rank, train.lora_rank),
        "optimizer": _or(declared.optimizer, train.optimizer),
        "loss_fn": loss_fn,
        "loss_fn_config": loss_fn_config,
        "max_staleness": declared.max_staleness,
        "load_optimizer_state": declared.load_optimizer_state,
        "batch_size": declared.batch_size,
        "checkpoint": declared.checkpoint,
    }


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


def trained_policies(config: Any) -> tuple[PolicyId, ...]:
    """The ids of the policies the run trains, in order: ``main``, then each ``policies:`` entry that is not frozen."""

    declared = _declared(config)
    return (MAIN, *(policy for policy, entry in declared.items() if getattr(entry, "trainable", True)))


def steps_independently(config: Any, stepping: Stepping | None = None) -> bool:
    """Whether the run steps its trained policies each on its own clock: its trainer's stepping is ``independent``
    (``stepping``, its ``needs().stepping``; None: ``train.stepping``, the rl trainer's answer, the only trainer a
    config selects in this release), and it trains two or more (with one, independent stepping is lockstep: there is
    one clock)."""

    stepping = config.train.stepping if stepping is None else stepping
    return stepping == "independent" and len(trained_policies(config)) > 1


def horizon_bound(config: Any) -> int | None:
    """The staleness bound the dispatcher's horizon counts (``k``: at most ``k + 1`` batches of groups started and not
    yet trained): the smallest bound of the trained policies that agents sample, each its own ``max_staleness``, else
    ``train.max_staleness`` (main's always); None when none is bounded.  With one policy, ``train.max_staleness``."""

    run, declared = config.train.max_staleness, _declared(config)
    sampled = agent_bindings(config).sampled()
    bounds = [
        run if policy == MAIN else _or(declared[policy].max_staleness, run)
        for policy in trained_policies(config)
        if policy in sampled
    ]
    bounded = [bound for bound in bounds if bound is not None]
    return min(bounded) if bounded else None


# A policy's id: a lowercase letter, then lowercase letters, digits, ``_`` and ``-``, 32 characters at most.
POLICY_ID = re.compile(r"[a-z][a-z0-9_-]{0,31}")
# The root agent's role (the gateway's AgentSession.role), whose agent samples agents.policy whatever agents.roles says.
ROOT_ROLE = "policy"
# The roles platoon's own code opens agents with: the root agent's, that of a child a harness spawns unless it names
# another (ctx.agents.spawn), a verifier's and a judge's.  A harness may mint agents of any other role.
PLATOON_ROLES: Final = (ROOT_ROLE, "subagent", VERIFIER_ROLE, JUDGE_ROLE)


def policy_problems(config: Any) -> list[str]:
    """What loading refuses of the config's ``policies:`` and ``agents:``, one entry each: an id that is no policy id
    (:data:`POLICY_ID`) or that names ``main``, which ``train:`` declares; a trained policy's keys on a frozen policy,
    and an optimizer state to load without the checkpoint it is in (:func:`_entry_problems`); an agent role, or the
    agents' default, bound to a policy the run does not declare; a role's ``tools`` that is no list of names or
    patterns; a frozen policy as the agents' default, which the root agent samples; and a policy the run trains that no
    agent samples, which no row would ever train (:func:`_unsampled_problems`)."""

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
        else:
            problems.extend(_entry_problems(where, policy))
    known = {MAIN, *declared}
    agents = getattr(config, "agents", None)
    default = getattr(agents, "policy", None)
    if default is not None and default not in known:
        problems.append(f"agents.policy: {default!r} is not a policy of the run ({', '.join(sorted(known))})")
    elif default is not None and default in declared and not getattr(declared[default], "trainable", True):
        problems.append(
            f"agents.policy: {default!r} is frozen, and the root agent samples a policy the run trains: name main or a "
            "policy the run trains, and bind a role to the frozen one (agents.roles.<role>.policy) instead"
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
    # Which policies the agents sample is read from bindings that the problems above may find wrong (a role bound to an
    # unknown policy, a frozen default): only once they hold does a policy no agent samples say something true.
    if problems:
        return problems
    return [*_unsampled_problems(config), *stepping_problems(config, independent=steps_independently(config))]


# The keys of a policies: entry that only a trained policy reads, and the value each has when it is left out.
_TRAINING_KEYS = {
    "lora_rank": None,
    "optimizer": None,
    "loss_fn": None,
    "loss_fn_config": None,
    "max_staleness": None,
    "load_optimizer_state": False,
    "batch_size": None,
    "checkpoint": None,
}
# The keys of a policies: entry that only a policy on its own clock reads (train.stepping: independent), each with the
# key of train: or checkpoint: that a policy stepping in lockstep follows instead.
_OWN_CLOCK_KEYS = {"batch_size": "train.batch_size", "checkpoint": "checkpoint.every and checkpoint.recovery_every"}


def _entry_problems(where: str, policy: Any) -> list[str]:
    """What loading refuses of one ``policies:`` entry (at ``where``): a frozen one's keys that only a trained policy
    reads, which it would ignore; and ``load_optimizer_state`` without ``load_checkpoint_path``, whose checkpoint
    holds the optimizer state."""

    problems = []
    if not getattr(policy, "trainable", True):
        if given := [key for key, unset in _TRAINING_KEYS.items() if getattr(policy, key, unset) != unset]:
            one = len(given) == 1
            problems.append(
                f"{where}: {', '.join(given)} {'is' if one else 'are'} for a policy the run trains, and this one is "
                f"frozen (trainable: false), so it never steps: leave {'it' if one else 'them'} out"
            )
    elif getattr(policy, "load_optimizer_state", False) and getattr(policy, "load_checkpoint_path", None) is None:
        problems.append(
            f"{where}.load_optimizer_state: it loads the optimizer state of the checkpoint "
            f"{where}.load_checkpoint_path names, and that is not set"
        )
    return problems


def _unsampled_problems(config: Any) -> list[str]:
    """A policy the run trains that no agent samples (``agents.policy``, ``agents.roles.<role>.policy``): no row would
    ever train it, yet every checkpoint would save it.  ``main`` is unsampled only beside an ``agents.policy`` that
    names another policy."""

    agents = agent_bindings(config)
    sampled = set(agents.sampled())
    problems = []
    if MAIN not in sampled:
        problems.append(
            f"agents.policy: the root agent samples {agents.default}, and no role samples main, the policy train: "
            "declares, which the run trains, so no row would ever train it: bind the agents of a role to main "
            "(agents.roles.<role>.policy: main)"
        )
    problems.extend(
        f"policies.{policy}: the run trains it, and no agent samples it, so no row would ever train it: bind the "
        f"agents of a role to it (agents.roles.<role>.policy: {policy}), or set policies.{policy}.trainable: false"
        for policy, declared in _declared(config).items()
        if getattr(declared, "trainable", True) and policy not in sampled
    )
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


def stepping_problems(config: Any, *, independent: bool) -> list[str]:
    """What the run refuses of how its trained policies step, ``independent``ly or together: a trained policy's own
    batch or checkpoint cadence when they step together, where it would be ignored; and, stepping independently, a
    batch that the step's minibatches and microbatches do not divide, and a group admission rule, whose steps under a
    cap this release counts for a run that steps together only.  Loading asks it of ``train.stepping``
    (:func:`steps_independently`), and the run again of its trainer's ``needs().stepping``."""

    train, problems = config.train, []
    parts = train.num_minibatches * train.num_microbatches
    for policy_id, declared in _declared(config).items():
        if not getattr(declared, "trainable", True):
            continue  # a frozen policy's training keys are refused with the others (_entry_problems)
        for key, instead in _OWN_CLOCK_KEYS.items():
            if getattr(declared, key, None) is not None and not independent:
                problems.append(
                    f"policies.{policy_id}.{key}: a policy's own {key.replace('_', ' ')} is for policies that step "
                    f"on their own clocks (train.stepping: independent); stepping together (in lockstep, as "
                    f"train.stepping or the trainer's needs() say), every trained policy follows {instead}: set "
                    "train.stepping: independent, or leave it out"
                )
        batch = getattr(declared, "batch_size", None)
        if independent and isinstance(batch, int) and batch % parts:
            problems.append(
                f"policies.{policy_id}.batch_size: {batch} does not split into train.num_minibatches × "
                f"train.num_microbatches ({parts}) parts: give a multiple of {parts}"
            )
    if independent:
        from platoon.train.admissions import rule_mode  # the rules' names, read at load only

        if rule_mode(config):
            problems.append(
                "train.stepping: independent steps each policy on its own clock, and a group admission rule other "
                "than all (train.group_admission, or an entry's) counts its groups toward the steps of a run that "
                "steps together: this release takes no rule under independent stepping; leave the rules at all, or "
                "step in lockstep"
            )
    return problems


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
    role that looks like a misspelling of one of :data:`PLATOON_ROLES` (:func:`_role_typos`); a role's tool pattern
    that matches no tool of the run's environments, when each declares its tools up front (:func:`_tool_warnings`);
    and, stepping independently, a trained policy that saves no full checkpoint before the run's end
    (:func:`_unsaved_policies`)."""

    agents = agent_bindings(config)
    used = set(agents.sampled())
    for environment in getattr(config, "environments", None) or ():
        used |= {value for value in (environment.credit_options or {}).values() if isinstance(value, str)}
    lines = [
        f"policies.{policy}: no agent samples it (agents.roles) and no entry's credit_options names it, so the run "
        "opens it and never uses it"
        for policy, declared in _declared(config).items()
        if policy not in used and not getattr(declared, "trainable", True)  # a trained one is refused (policy_problems)
    ]
    if agents.roles.get(ROOT_ROLE, agents.default) != agents.default:
        lines.append(
            f"agents.roles.{ROOT_ROLE}.policy: the root agent's role is {ROOT_ROLE}, and the root samples "
            f"agents.policy ({agents.default}) whatever its role: only a subagent minted with the role {ROOT_ROLE} "
            "samples this policy"
        )
    return [*lines, *_role_typos(config), *_tool_warnings(config), *_unsaved_policies(config)]


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


def _unsaved_policies(config: Any) -> list[str]:
    """Stepping independently, each trained policy that saves no full checkpoint before the run's end: no permanent
    cadence (``checkpoint.every``, its own or the run's, is 0) and no ``recovery_every``.  Each group whose share it
    consumed stays kept until then, and every other policy's commit writes them all again."""

    if not steps_independently(config):
        return []
    checkpoint = config.checkpoint
    lines = []
    if not checkpoint.every and checkpoint.recovery_every is None:
        lines.append(
            f"{MAIN} saves no full checkpoint before the run's end (checkpoint.every is 0 and "
            "checkpoint.recovery_every unset), and its policies step independently: each group it consumed stays kept "
            "until then, and every commit writes them all; set checkpoint.recovery_every"
        )
    for policy_id, declared in _declared(config).items():
        own = getattr(declared, "checkpoint", None)
        every, recovery = checkpoint.every, checkpoint.recovery_every
        if own is not None:  # its own cadence, each key left out the run's
            every = every if own.every is None else own.every
            recovery = recovery if own.recovery_every is None else own.recovery_every
        if getattr(declared, "trainable", True) and not every and recovery is None:
            lines.append(
                f"policies.{policy_id} saves no full checkpoint before the run's end (its checkpoint.every, or the "
                "run's, is 0, and no recovery_every is set), and the policies step independently: each group it "
                "consumed stays kept until then, and every commit writes them all; set "
                f"policies.{policy_id}.checkpoint.recovery_every"
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
