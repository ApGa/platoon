"""Reference harnesses for checking a dataset rather than a policy: ``platoon-oracle`` runs each task's reference
solution, and ``platoon-noop`` does nothing.  Neither makes a model call.

A sound task scores 1 with the oracle and 0 with the no-op, and its infrastructure fails rarely.  Run both on a
sample of a dataset before training on it, with the environment config training will use: they go through the
same rollout path as a trained harness (the env session, its sandbox and image, its grading, the outcome rules),
so the check covers what training runs.

``platoon-oracle`` needs an env session that can run its task's reference solution (``run_solution()``): a
Harbor session uploads the task's ``solution/`` to ``/solution`` and runs ``bash /solution/solve.sh`` in the task
sandbox, as Harbor's own oracle agent does.  As in Harbor, the session is graded whatever the script's exit code;
its exit code and the end of its output are recorded in the trajectory's misc under ``oracle``.  A script that
runs past the task's agent timeout ends the agent with ``timeout``, which the environment's ``on_harness_exit``
decides about like any other harness's (by default it is never graded).
"""

from __future__ import annotations

from platoon.harness.agent import AgentContext, AgentResult
from platoon.harness.function import harness

ORACLE_MISC_KEY = "oracle"
_OUTPUT_TAIL_CHARS = 2000


@harness("platoon-oracle")
async def oracle(ctx: AgentContext) -> AgentResult:
    """Run the task's reference solution in its sandbox (Harbor: ``solution/solve.sh``), then let the env grade."""

    run_solution = getattr(ctx.session, "run_solution", None)
    if run_solution is None:
        return AgentResult(
            "error",
            error=f"platoon-oracle needs an env session with a reference solution; {type(ctx.session).__name__} "
            "has none (Harbor sessions run solution/solve.sh)",
        )
    result = await run_solution()
    output = (result.stdout + result.stderr)[-_OUTPUT_TAIL_CHARS:]
    ctx.annotate(**{ORACLE_MISC_KEY: {"exit_code": result.exit_code, "timed_out": result.timed_out, "output": output}})
    if result.timed_out:
        return AgentResult("timeout", error="the reference solution timed out")
    return AgentResult("completed", final_message=f"solve.sh exited with {result.exit_code}")


@harness("platoon-noop")
async def noop(ctx: AgentContext) -> AgentResult:
    """Do nothing, so the env grades the task as it started: a sound task scores 0."""

    return AgentResult("completed")
