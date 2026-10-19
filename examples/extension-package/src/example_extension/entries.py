"""``example-arithmetic@1``: the package's named environment.

It says once how to run the arithmetic tasks: the task table, the harness that plays them and its options, the
program that grades them, the reward processor that trains on them, and the splits. A config selects it with
``env: example-arithmetic@1`` and may override any of its keys.
"""

from __future__ import annotations

import platoon as pt

ARITHMETIC = pt.EnvironmentEntry(
    "example-arithmetic",
    version="1",
    description="Arithmetic questions with numeric answers, graded by exact match (a task table)",
    service="example_extension.tasks:arithmetic",
    harness="example-self-check",
    harness_defaults={"example-self-check": {"checks": 1}},
    program="example-exact-answer",
    reward_processor="example-partial-credit",
    dataset_kwargs={"split_name": "train"},
    eval_dataset_kwargs={"split_name": "test"},
)
