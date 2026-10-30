"""Which trainer a run trains with: the config's ``trainer:``, one of core's trainers (:data:`CORE_TRAINERS`), with its
``trainer_options``; checked when the config loads (:func:`trainer_problems`, exit 78 before anything runs).

This release has two trainers: ``rl`` (the default), GRPO-style training on the rollouts of the run's tasks, whose
settings are ``train:``'s and whose one option is the step hooks it attaches to each of its steps (:class:`RLOptions`);
and ``sft``, supervised fine-tuning on datasets, whose settings are ``trainer_options``.  ``interleaved`` is reserved
for a trainer platoon adds; a trainer that a package writes (RFC 0021) is selectable once platoon resolves those names.
Under a trainer other than ``rl``, a ``train:`` key that only ``rl`` reads
(:data:`~platoon.train.config_defs.RL_ONLY_TRAIN_KEYS`) is refused when set, with the option to use instead, and so are
the run's other policies and the agents' bindings (``policies:``, ``agents:``), which only ``rl``'s rollouts and credit
estimators use, and evaluation during training, which only ``rl`` runs so far; and the data such a trainer trains on
is checked as the config loads, as far as it can be without loading it.  The options that name files (an options class's
``path_options``) are relative to the config file that names them (:func:`trainer_path_options`).  Imports no Tinker.
"""

import dataclasses
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from platoon.options import schema_options
from platoon.train.trainer.hook_config import HookSpec, hook_specs

if TYPE_CHECKING:
    from platoon.train.trainer.api import Trainer


@dataclass(frozen=True)
class CoreTrainer:
    """One of core's trainers: its class and its options' class (None: it takes none), both as ``module:attr``, the
    options checked field by field, or by the class's own ``parse`` when it has one (rl's, whose hooks name classes);
    whether it rolls out the run's tasks, as its class's ``needs()`` says too; and for one that does not, the functions
    of its parsed options, as ``module:function``, that check without Tinker what it trains on instead:
    ``data_problems``, what the config's load finds wrong with it, and ``data_listing``, what ``platoon config validate
    --tasks`` lists of it (and raises ``ConfigError`` for)."""

    cls: str
    options: str | None
    rolls_out: bool
    data_problems: str | None = None
    data_listing: str | None = None


# Core's trainers, by the name a config gives them.
CORE_TRAINERS = {
    "rl": CoreTrainer(
        "platoon.train.trainer.rl:PlatoonTinkerRLTrainer",
        options="platoon.train.trainer.registry:RLOptions",  # its step hooks, read by RLOptions.parse
        rolls_out=True,
    ),
    "sft": CoreTrainer(
        "platoon.train.trainer.sft:SupervisedTrainer",
        options="platoon.train.supervised.options:SupervisedOptions",
        rolls_out=False,
        data_problems="platoon.train.supervised.registry:entry_problems",  # each dataset's entry in dataset_info.json
        data_listing="platoon.train.supervised.loading:dataset_lines",  # each dataset's conversations, loaded
    ),
}
# Core's names of the trainers platoon adds: reserved, so that no config or package takes them meanwhile.
RESERVED_TRAINERS = ("interleaved",)


@dataclass(frozen=True)
class RLOptions:
    """The rl trainer's options (``trainer_options``): the step hooks it attaches to each of its steps (``hooks``, RFC
    0021 section 9: :mod:`platoon.train.trainer.hook_config`).  Its other settings are ``train:``'s."""

    # The step hooks it attaches to each of its steps, each ``pkg.mod:Hook`` (or a name a package registered) or
    # ``{hook: pkg.mod:Hook, options: {...}}`` (docs/extending/step-hooks.md).
    hooks: tuple[HookSpec, ...] = ()

    @classmethod
    def parse(cls, values: Mapping[str, Any]) -> "RLOptions":
        """The options of ``values``, ``trainer_options`` as the config gives them; a ValueError names what is
        wrong."""

        if unknown := sorted(set(values) - {"hooks"}):
            raise ValueError(f"the rl trainer takes hooks, not {', '.join(unknown)}: its settings are train:'s")
        return cls(hooks=hook_specs(values.get("hooks") or []))


def trainer_problems(config: Any) -> list[str]:
    """What the config's ``trainer:`` and ``trainer_options`` refuse, one entry each: a name that is not one of
    :data:`CORE_TRAINERS`; options its schema does not take; under a trainer other than ``rl``, the ``train:`` keys it
    would not read, ``policies:``, ``agents:`` and evaluation during training; and once all that holds, what the
    trainer's ``data_problems`` find (sft: a dataset that ``dataset_info.json`` does not name, or names in a way this
    release does not read)."""

    name = getattr(config, "trainer", "rl")
    if name not in CORE_TRAINERS:
        trainers = " and ".join(CORE_TRAINERS)
        if name in RESERVED_TRAINERS:
            return [f"trainer: {name!r} is reserved for a trainer platoon will add; this release has {trainers}"]
        return [f"trainer: {name!r} is not a trainer: this release has {trainers} (rl, the default)"]
    problems = _options_schema(name).option_problems(getattr(config, "trainer_options", None), "trainer_options")
    if name != "rl":
        problems += _unread_settings(config, name)
    if not problems and (check := CORE_TRAINERS[name].data_problems) is not None:
        problems += _imported(check)(options_of(config))
    return problems


def rolls_out(config: Any) -> bool:
    """Whether the config's trainer rolls out the run's tasks, which rollouts then sample the policy's checkpoints
    from: ``rl`` does, ``sft`` does not."""

    core = CORE_TRAINERS.get(getattr(config, "trainer", "rl"))
    return core is None or core.rolls_out


def data_listing(config: Any) -> list[str]:
    """What ``platoon config validate --tasks`` lists of the data the config's trainer trains on instead of rollouts,
    one line each; raises :class:`~platoon.utils.config.ConfigError` for what it cannot read."""

    listing = CORE_TRAINERS[getattr(config, "trainer", "rl")].data_listing
    return _imported(listing)(options_of(config)) if listing is not None else []


def trainer_path_options(name: Any) -> dict[str, str | None]:
    """The ``trainer_options`` of trainer ``name`` that name files (its options class's ``path_options``), each with
    its default (None: it has none); for a name that is not one of :data:`CORE_TRAINERS`, such as that of a config file
    that leaves ``trainer:`` to the file it extends, those of every core trainer."""

    names = [name] if name in CORE_TRAINERS else list(CORE_TRAINERS)
    found: dict[str, str | None] = {}
    for each in names:
        reference = CORE_TRAINERS[each].options
        schema = _imported(reference) if reference is not None else None
        defaults = {item.name: item.default for item in dataclasses.fields(schema)} if schema is not None else {}
        for option in getattr(schema, "path_options", ()):
            default = defaults.get(option)
            found[option] = default if isinstance(default, str) else None
    return found


def trainer_class(config: Any) -> "type[Trainer]":
    """The class of the config's trainer, imported."""

    return _imported(CORE_TRAINERS[getattr(config, "trainer", "rl")].cls)


def trainer_options(values: Mapping[str, Any] | None) -> RLOptions:
    """``values``, the rl trainer's ``trainer_options`` as a config gives them, parsed (a ``ConfigError`` with each
    problem)."""

    return _options_schema("rl").parse(values, "trainer_options")


def options_of(config: Any) -> Any:
    """The config's ``trainer_options``, parsed by its trainer's schema."""

    name = getattr(config, "trainer", "rl")
    return _options_schema(name).parse(getattr(config, "trainer_options", None), "trainer_options")


def lock_record(config: Any) -> dict[str, Any]:
    """What ``lock.json`` records of the run's trainer: its name, and its options with their defaults."""

    schema = _options_schema(getattr(config, "trainer", "rl"))
    return {"name": config.trainer, "options": schema.as_json(options_of(config))}


def _options_schema(name: str) -> Any:
    """The option schema of core's trainer ``name`` (:func:`platoon.options.schema_options`): its options class, or
    the class's own parser (``parse``) when it has one."""

    reference = CORE_TRAINERS[name].options
    options = _imported(reference) if reference is not None else None
    return schema_options(getattr(options, "parse", options), name=f"trainer: {name}")


def _unread_settings(config: Any, name: str) -> list[str]:
    """The settings a trainer other than ``rl`` would not read, one entry each: a ``train:`` key that only ``rl``
    reads, set to other than its default (with the trainer's option of the same name, when it has one); the run's
    other policies (``policies:``) and the agents' bindings (``agents:``), which only ``rl`` samples or scores with;
    and evaluation during training (``eval.every``)."""

    from platoon.train.config_defs import RL_ONLY_TRAIN_KEYS, TrainConfig, evaluates

    train = config.train
    default = TrainConfig(model_name=train.model_name, renderer_name=train.renderer_name)
    # Every rollout of a batch by default, which follows the batch's own settings.
    default.max_concurrent_rollouts = train.batch_size * train.workflow_config.group_size
    options = CORE_TRAINERS[name].options
    option_names = {item.name for item in dataclasses.fields(_imported(options))} if options is not None else set()
    problems = []
    for key in RL_ONLY_TRAIN_KEYS:
        if getattr(train, key) == getattr(default, key):
            continue
        instead = f"set trainer_options.{key}" if key in option_names else "leave it out"
        problems.append(f"train.{key}: only the rl trainer reads it, and this run trains with {name}: {instead}")
    if getattr(config, "policies", None):
        problems.append(
            f"policies: only the rl trainer samples or scores with other policies, and this run trains with {name}: "
            "leave it out"
        )
    agents = getattr(config, "agents", None)
    if getattr(agents, "policy", None) is not None or getattr(agents, "roles", None):
        problems.append(
            f"agents: only the rl trainer's agents sample policies, and this run trains with {name}: leave it out"
        )
    if evaluates(config):
        problems.append(
            f"eval.every: trainer: {name} does not evaluate during training yet: set eval.every: 0, and evaluate the "
            "run's checkpoints with platoon eval"
        )
    return problems


def _imported(reference: str) -> Any:
    module, _, attr = reference.partition(":")
    return getattr(importlib.import_module(module), attr)
