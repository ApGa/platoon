import asyncio
import traceback

from platoon.agents.base import Agent
from platoon.envs.base import Env, Observation
from platoon.episode.context import (
    budget_tracker,
    current_agent,
    current_env,
    current_trajectory,
    current_trajectory_collection,
    episode_step_timeout,
    error_message,
    finish_message,
)
from platoon.episode.trajectory import StepBudgetTracker, Trajectory, TrajectoryCollection
from platoon.utils.span_profile import profile_span


# NOTE: Call using asyncio.create_task() to make sure edits to contextvars do not leak to parent context
async def run_episode(agent: Agent, env: Env, verbose: bool = False, timeout: int | None = 300) -> Trajectory:
    try:
        step_count = 0
        set_context_vars(agent, env, timeout=timeout)
        traj = current_trajectory.get()
        async with profile_span(
            "run_episode",
            metadata={
                "agent_type": type(agent).__name__,
                "env_type": type(env).__name__,
                "task_id": getattr(env.task, "id", None),
                "timeout": timeout,
                "trajectory_id": traj.id,
                "parent_trajectory_id": traj.parent_info.id if traj.parent_info is not None else None,
            },
        ):
            obs = await env.reset()
            while not halt_episode(obs):
                action = await asyncio.wait_for(agent.act(obs), timeout=timeout)
                obs = await asyncio.wait_for(env.step(action), timeout=timeout)
                step_count += 1
    except (Exception, asyncio.CancelledError) as e:
        tb_summary = traceback.extract_tb(e.__traceback__)
        origin = ""
        if tb_summary:
            last_frame = tb_summary[-1]
            origin = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        detailed_msg = (
            f"Error in episode loop at step {step_count}"
            + (f" ({origin})" if origin else "")
            + f"\n{e.__class__.__name__}: {e}\n"
            + traceback.format_exc()
        )
        if verbose:
            print(detailed_msg)
        error_message.set(detailed_msg)
    finally:
        try:
            await agent.close()
        except BaseException:
            pass
        try:
            await env.close()
        except BaseException:
            pass
        # Finalize trajectory and emit a finish event to sinks
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj.error_message = error_message.get()
        traj.finish_message = finish_message.get()
        # TODO: We could move trajectory finish logic (rewards, finish message, etc.) from env to here.
        traj_collection.finish_trajectory(traj.id)
        return traj


def set_context_vars(agent: Agent, env: Env, timeout: int | None):
    finish_message.set(None)
    error_message.set(None)
    episode_step_timeout.set(timeout)
    current_agent.set(agent)
    current_env.set(env)

    if current_trajectory_collection.get(None) is None:
        current_trajectory_collection.set(TrajectoryCollection())

    parent_traj = current_trajectory.get(None)
    current_trajectory.set(current_trajectory_collection.get().create_trajectory(parent_traj=parent_traj))

    if budget_tracker.get(None) is None:
        budget_tracker.set(StepBudgetTracker())


def halt_episode(obs: Observation) -> bool:
    exhausted_budget = budget_tracker.get().remaining_budget() <= 0
    if exhausted_budget:
        error_message.set("WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete.")
    if finish_message.get(None) is not None:
        obs.finished = True
    return obs.finished or exhausted_budget
