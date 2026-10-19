"""``eval-submitter``: registered by importing this module, as a project's ``package:`` module registers it."""

from __future__ import annotations

import platoon as pt


@pt.harness("eval-submitter")
async def submitter(ctx: pt.AgentContext) -> pt.AgentResult:
    """Submit at once, without a model call."""
    result = await ctx.call_tool("submit", {})
    return pt.AgentResult(final_message=result.as_text())
