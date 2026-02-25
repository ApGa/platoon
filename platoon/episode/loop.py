import asyncio
import traceback

from platoon.episode.trajectory import StepBudgetTracker, Trajectory, TrajectoryCollection
from platoon.agents.base import Agent
from platoon.envs.base import Env, Observation
from platoon.episode.context import (
    current_agent,
    current_env,
    current_trajectory,
    current_trajectory_collection,
    error_message,
    budget_tracker,
    finish_message,
)

CLEANUP_TIMEOUT = 180  # seconds to allow each close() call before giving up
# from openhands.sdk.conversation import ConversationExecutionStatus
# from platoon.utils.openhands_utils import is_finished
# def agent_finished(obs):
#     if obs.conversation_state.execution_status in [
#         ConversationExecutionStatus.FINISHED,
#         ConversationExecutionStatus.STUCK,
#         ConversationExecutionStatus.ERROR
#     ]:
#         return True
#     return False

# NOTE: This function should be called using asyncio.create_task() to make sure edits to contextvars do not leak to parent context
async def run_episode(agent: Agent, env: Env, verbose: bool = True, timeout: int = 300) -> Trajectory:
    try:
        step_count = 0
        set_context_vars(agent, env)
        obs = await env.reset()
        # while True:
        #     import time
        #     time.sleep(10000000)
        while not halt_episode(obs):
            # if agent_finished(obs):
            #     print("OpenHands Finished -- waiting for agent.act() to complete", flush=True)
            action = await asyncio.wait_for(agent.act(obs), timeout=timeout)
            # if agent_finished(obs):
            #     print("OpenHands Finished -- waiting for env.step() to complete", flush=True)
            obs = await asyncio.wait_for(env.step(action), timeout=timeout)
            # if agent_finished(obs):
            #     print("OpenHands Finished -- env.step() completed", flush=True)
            #     if not is_finished(obs):
            #         print(f"WARNING: Conversation execution status is {obs.conversation_state.execution_status} but is_finished() returned False", flush=True)
            step_count += 1
    except asyncio.CancelledError:
        # Task was cancelled by parent (e.g. rollout timeout via wait_for).
        # Catch it so the finally block can run normally without re-cancellation.
        error_message.set(f"Episode cancelled at step {step_count} (likely rollout timeout)")
        if verbose:
            print(f"Episode cancelled at step {step_count}", flush=True)
    except Exception as e:
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
        # Cleanup with bounded timeouts so a blocking close() can't stall the process.
        # Use asyncio.shield() so that CancelledError from a parent wait_for()
        # doesn't prevent cleanup from running.
        for label, closeable in [("agent", agent), ("env", env)]:
            try:
                await asyncio.shield(
                    asyncio.wait_for(closeable.close(), timeout=CLEANUP_TIMEOUT)
                )
            except asyncio.CancelledError:
                print(f"Warning: {label}.close() was cancelled, cleanup may be incomplete", flush=True)
            except asyncio.TimeoutError:
                print(f"Warning: {label}.close() timed out after {CLEANUP_TIMEOUT}s, skipping", flush=True)
            except Exception as e:
                print(f"Warning: {label}.close() raised {e}, skipping", flush=True)
        # Finalize trajectory and emit a finish event to sinks
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj.error_message = error_message.get()
        traj.finish_message = finish_message.get()
        if traj.finish_message is None:
            traj.finish_message = "Episode finished without a finish message."
        if traj.error_message is None:
            traj.error_message = "Rollout finished without an error or finish message"
        # TODO: We could move out trajectory finish logic (adding up rewards, setting finish message, etc.) from env logic to here.
        traj_collection.finish_trajectory(traj.id)
        return traj

def set_context_vars(agent: Agent, env: Env):
    finish_message.set(None)
    error_message.set(None)
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
        error_message.set(f"WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete.")
    if finish_message.get(None) is not None:
        obs.finished = True
    return obs.finished or exhausted_budget
