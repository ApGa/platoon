"""The training and evaluation workflows ``platoon train`` builds for a run's config (:func:`built_workflows`), for
the tests of what the config's settings give the workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from platoon.train import entry_settings, launch
from platoon.train.auto import AutoEnvironment
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec


def built_workflows(config: Any) -> tuple[Any, Any]:
    """The training and evaluation workflows ``platoon train`` builds for ``config``, on its policy main of a model
    that nothing loads."""

    AutoEnvironment.load(config)
    settings = entry_settings.entry_settings(config)
    components = launch._build_components(config, AutoEnvironment.from_config(config))
    model_info: Any = SimpleNamespace(
        model_name="m", renderer_name="r", renderer_kwargs={}, llm=SimpleNamespace(version=0)
    )
    policies = PolicySet([Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), model_info)])
    training: Any = SimpleNamespace(tasks=[])
    return launch._build_workflows(config, components, policies, settings, training, None)
