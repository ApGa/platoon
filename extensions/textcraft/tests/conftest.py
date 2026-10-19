"""The TextCraft extension's tests share core's test helpers and settings, from the repository's ``tests/``: the
suite's settings (``suite_settings``), the built-in harnesses' fixtures (``builtin_fixtures``), the inline runner
(``inline_runs``), the golden-fixture helpers (``codeact_golden``, ``golden_scenarios``) and the scripted agent trees
(``scripted_trees``).  Only first-party extensions do this; a third party tests with ``platoon.testing``.
"""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3] / "tests"
for helpers in (TESTS / "envs", TESTS / "harness" / "builtin", TESTS):
    if str(helpers) not in sys.path:
        sys.path.insert(0, str(helpers))

from builtin_fixtures import *  # noqa: E402, F403
from suite_settings import *  # noqa: E402, F403
