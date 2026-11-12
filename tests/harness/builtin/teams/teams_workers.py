"""What a rollout worker imports by name (``rollout_isolation: process``) in ``test_teams_isolation.py``, with this
directory and core's ``tests/testing`` on its path: a rollout of a team by the ``platoon-harness`` program, sampling a
scripted policy of its own, as the worker's policy would be."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from teams_toys import SOLO, rollout, session_name

from platoon.train.tinker.proxy import proxy_interactions

# The directory each rollout notes the process it ran in, and its session, by name.
FACTS = "TEAMS_WORKER_FACTS"


async def team_rollout(task: Any, config: Any) -> dict[str, Any]:
    """One rollout of the lockbox by the team ``SOLO``, whose calls the worker's own record of the rollout's model calls
    then holds, as it holds those its policy samples; it notes where it ran."""

    run = await rollout(SOLO)
    proxy_interactions.get().update(run.interactions)
    session = session_name(run)
    Path(os.environ[FACTS], f"{session}.json").write_text(json.dumps({"pid": os.getpid(), "session": session}))
    return run.collection.to_dict()
