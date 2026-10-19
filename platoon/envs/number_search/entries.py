"""Number search as a named environment: ``env: platoon-number-search@1`` in an ``environments`` entry.

A built-in: it ships with platoon and needs nothing installed.  Core registers it by name
(``platoon.envs.named.BUILTIN_ENVIRONMENTS``) and imports this module only when a config names it.  The entry runs the
ORS environment in process, gives ``platoon-codeact`` the number-search prompts (``prompts.py``) and
``platoon-react`` a system prompt, and evaluates on the ``val`` split::

    environments:
      - env: platoon-number-search@1
        harness: platoon-codeact
        dataset_kwargs: {limit: 1000}
"""

from __future__ import annotations

import platoon as pt

NUMBER_SEARCH = pt.EnvironmentEntry(
    "platoon-number-search",
    version="1",
    description="Guess a hidden integer in a range, told 'too low' or 'too high' after each guess (ORS, in process)",
    service="python://platoon.envs.number_search.env:NumberSearchEnvironment",
    harness_defaults={
        "platoon-codeact": {
            "prompt_builder": "platoon.envs.number_search.prompts:NumberSearchPromptBuilder",
            "include_reasoning": False,
            "subagents": False,
        },
        "platoon-react": {
            "system_prompt": (
                "Find the hidden number. Call the guess tool with one integer per turn; it tells you whether your "
                "guess is too low, too high or correct."
            ),
        },
    },
    agents={"max_depth": 0},
    eval_dataset_kwargs={"split_name": "val"},
)
