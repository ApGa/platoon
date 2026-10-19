"""Toys for the platoon.testing tests, defined as an extension's own tests would define or import theirs: an ORS
environment (``python://testing_toys:Lockbox``), a function harness that can delegate, and an episode program."""

from __future__ import annotations

from openreward.environments import Environment, TextBlock, ToolOutput, tool
from pydantic import BaseModel

import platoon as pt

LOCKBOX = "python://testing_toys:Lockbox"


class Code(BaseModel):
    code: str


class Lockbox(Environment):
    """Unlock the box with the code the prompt gives; a wrong code is answered and the episode goes on."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"id": f"box-{n}", "code": str(n)} for n in range(3)]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=f"Open the box. Its code is {self.task_spec['code']}.")]

    @tool
    async def unlock(self, args: Code) -> ToolOutput:
        """Try a code."""

        if args.code == self.task_spec["code"]:
            return ToolOutput(blocks=[TextBlock(text="unlocked")], reward=1.0, finished=True)
        return ToolOutput(blocks=[TextBlock(text="wrong code")])


@pt.harness("testing-delegator")
async def delegator(ctx: pt.AgentContext) -> pt.AgentResult:
    """Asks the model once: ``DELEGATE`` hands the task to a subagent, anything else is tried as the code."""

    reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])
    if reply.text.startswith("DELEGATE"):
        child = await ctx.agents.spawn("Find the code and unlock the box.", max_steps=2)
        return pt.AgentResult("completed", final_message=child.message)
    result = await ctx.call_tool("unlock", {"code": reply.text.strip()})
    return pt.AgentResult("completed", final_message=result.as_text())


@pt.program("testing-tagged")
async def tagged(ep: pt.EpisodeContext) -> pt.Outcome:
    """Runs the root agent, grades it, and tags every trajectory once the collection is built."""

    def tag(collection: pt.TrajectoryCollection) -> None:
        for trajectory in collection.trajectories.values():
            trajectory.misc["tagged"] = True

    await ep.run()
    ep.add_finalizer(tag)
    return await ep.grade()
