"""TextCraft and TextCraft-Synth as named environments: ``env: textcraft@1`` or ``env: textcraft-synth@1``.

This distribution registers them in the ``platoon.environments`` entry-point group (``pyproject.toml``), so
platoon imports this module only when a config names one.  Each runs its ORS environment in process, gives
``platoon-codeact`` the TextCraft prompts (``prompts.py``; a config picks the arm with
``prompt_builder_options.variant``), and evaluates on the ``val`` split; TextCraft-Synth also scores with core's
RAO reward processor, ``platoon-delegation``.  The default program, ``single``, applies the rollout config's
``skip_subagent_reward_computation`` and ``propagate_root_success``::

    environments:
      - env: textcraft-synth@1
        harness: platoon-codeact
        harness_options:
          prompt_builder_options: {variant: recursive}
        agents: {max_depth: null, budget: linear, child_grading: env}
        dataset_kwargs: {split_name: train-medium}
"""

from __future__ import annotations

import platoon as pt

PROMPT_BUILDER = "platoon_textcraft.prompts:TextCraftPromptBuilder"

TEXTCRAFT = pt.EnvironmentEntry(
    "textcraft",
    version="1",
    description="Craft Minecraft items by recipe with craft, get_info, view_inventory and finish (ORS, in process)",
    service="python://platoon_textcraft.env:TextCraftEnvironment",
    harness_defaults={
        "platoon-codeact": {"prompt_builder": PROMPT_BUILDER, "prompt_builder_options": {"synth": False}}
    },
    eval_dataset_kwargs={"split_name": "val"},
)

TEXTCRAFT_SYNTH = pt.EnvironmentEntry(
    "textcraft-synth",
    version="1",
    description=(
        "TextCraft on synthetic recipes with deeper crafting trees and easy to extreme splits (ORS, in process)"
    ),
    service="python://platoon_textcraft.env:TextCraftSynthEnvironment",
    reward_processor="platoon-delegation",
    harness_defaults={"platoon-codeact": {"prompt_builder": PROMPT_BUILDER, "prompt_builder_options": {"synth": True}}},
    eval_dataset_kwargs={"split_name": "val"},
)
