"""``eval-program``: registered by importing this module, as a project's ``package:`` module registers it."""

from __future__ import annotations

import platoon as pt


@pt.program("eval-program")
async def run_then_grade(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()
