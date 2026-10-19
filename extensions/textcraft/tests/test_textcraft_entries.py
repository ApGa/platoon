"""TextCraft's named environments, from the installed extension's metadata: ``textcraft@1`` and
``textcraft-synth@1`` resolve through their entry points, pass ``platoon extensions check``, and stand for the
environment entries that configs spelled out before they could name them."""

from __future__ import annotations

from importlib.metadata import distribution

import pytest
from platoon import extensions
from platoon.envs.named import expand_environment, get_environment

DISTRIBUTION = "platoon-textcraft"


@pytest.mark.parametrize("ref", ["textcraft@1", "textcraft-synth@1"])
def test_the_named_environments_resolve_from_the_installed_entry_points(ref):
    name = ref.partition("@")[0]

    entry = get_environment(name)

    assert entry.ref == ref
    origin = extensions.recorded_origin("environment", ref)
    assert origin is not None
    assert (origin.distribution, origin.version, origin.entry_point) == (
        DISTRIBUTION,
        distribution(DISTRIBUTION).version,
        f"platoon.environments:{name}",
    )
    expanded = expand_environment({"env": ref, "harness": "platoon-codeact"})
    for target in (expanded["service"].removeprefix("python://"), expanded["harness_options"]["prompt_builder"]):
        module, _, attribute = target.partition(":")
        assert hasattr(__import__(module, fromlist=[attribute]), attribute), target


def test_the_extension_passes_platoons_checks():
    assert extensions.check_extensions([DISTRIBUTION], load=True) == []


def test_textcraft_synth_by_name_is_the_entry_its_configs_spelled_out():
    """``env: textcraft-synth@1`` with the linear RAO arm's own keys: the environments entry that arm spelled out."""

    named = {
        "env": "textcraft-synth@1",
        "harness": "platoon-codeact",
        "harness_options": {
            "prompt_builder_options": {"variant": "linear"},
            "subagents": False,
            "cell_timeout_s": 7200,
        },
        "agents": {"max_depth": 0},
        "dataset_kwargs": {"split_name": "train-medium"},
        "eval_dataset_kwargs": {"limit": 100},
    }
    spelled_out = {
        "service": "python://platoon_textcraft.env:TextCraftSynthEnvironment",
        "harness": "platoon-codeact",
        "harness_options": {
            "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
            "prompt_builder_options": {"variant": "linear", "synth": True},
            "subagents": False,
            "cell_timeout_s": 7200,
        },
        "agents": {"max_depth": 0},
        "reward_processor": "platoon-delegation",
        "dataset_kwargs": {"split_name": "train-medium"},
        "eval_dataset_kwargs": {"split_name": "val", "limit": 100},
    }

    expanded = expand_environment(named)

    assert expanded.pop("env") == "textcraft-synth@1" and expanded == spelled_out
