"""Number search, platoon's built-in example environment: guess a hidden integer, told "too low" or "too high" after
each guess.

``env`` is the ORS environment (openreward and pydantic only), ``tasks`` generates its tasks from a seed, ``prompts``
holds the CodeAct prompts of the earlier number-search agent, and ``entries`` the named environment
``platoon-number-search@1``, which core registers as a built-in (:mod:`platoon.envs.named`).  This module imports none
of them, so a run loads only what it names.
"""
