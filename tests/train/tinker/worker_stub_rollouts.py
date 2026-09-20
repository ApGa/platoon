"""Rollout functions used by the real worker entrypoint's tests.

Importable by name from inside a worker process (the test puts this directory
on the child's ``PYTHONPATH``), so the worker resolves them exactly the way it
resolves a production ``rollout_fn``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time


def _write_facts() -> None:
    """Record what the worker's environment actually looks like from inside."""

    path = os.environ.get("STUB_FACTS_PATH")
    if not path:
        return
    import torch

    facts = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "is_session_leader": os.getpgid(os.getpid()) == os.getpid(),
        "TINKER_TELEMETRY": os.environ.get("TINKER_TELEMETRY"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "PLATOON_TINKER_ROLLOUT_WORKER": os.environ.get("PLATOON_TINKER_ROLLOUT_WORKER"),
        "subprocess_sampling_present": "TINKER_SUBPROCESS_SAMPLING" in os.environ,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "request_file_exists": os.path.exists(os.environ.get("STUB_REQUEST_PATH", "")),
        "extra_allow": _extra_allow(),
        "cwd": os.getcwd(),
    }
    with open(path, "w") as handle:
        json.dump(facts, handle)


def _extra_allow() -> bool:
    try:
        import tinker

        return tinker.types.SampleResponse.model_config.get("extra") == "allow"
    except Exception:
        return False


async def report_and_finish(_task, _config) -> dict:
    _write_facts()
    return {"trajectories": {}}


async def spin_forever(_task, _config) -> dict:
    _write_facts()
    # A pure-Python spin: the interpreter still runs signal handlers at
    # bytecode boundaries, so this is killed by the worker's own SIGALRM.
    while True:
        pass


async def sleep_forever(_task, _config) -> dict:
    _write_facts()
    await asyncio.sleep(1e9)
    return {"trajectories": {}}


async def slow_but_finite(_task, _config) -> dict:
    _write_facts()
    time.sleep(float(os.environ.get("STUB_SLEEP_SECONDS", "1")))
    return {"trajectories": {}}


def _claim_slot(output_dir: str, slots: int = 8) -> int:
    """Take a distinct index for this rollout, whichever process runs it.

    Group members are handed identical tasks and configs, so a differing
    reward has to come from somewhere shared.  An exclusive-create race over a
    fixed set of files gives every member of a group a different index in both
    isolation modes, which is what makes the two modes comparable.
    """

    os.makedirs(output_dir, exist_ok=True)
    for index in range(slots):
        try:
            descriptor = os.open(
                os.path.join(output_dir, f"slot-{index}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            continue
        os.close(descriptor)
        return index
    return slots


async def varied_reward_rollout(task, config) -> dict:
    """A real, convertible trajectory with a per-member reward.

    Registers one proxy interaction and one step that references it, so the
    production converter produces a genuine ``tinker.Datum`` -- including its
    routing handle -- in whichever process runs this.
    """

    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs

    from platoon.train.tinker.proxy import TinkerLLMInteraction, proxy_interactions

    index = _claim_slot(os.environ.get("STUB_SLOT_DIR") or config.output_dir)
    completion_id = f"completion-{index}"
    proxy_interactions.get()[completion_id] = TinkerLLMInteraction(
        obs=tinker.ModelInput.from_ints([11, 12, 13, 14]),
        action=TokensWithLogprobs(tokens=[15, 16], maybe_logprobs=[-0.25, -0.5]),
        routing_id=1000 + index,
    )
    return {
        "trajectories": {
            "root": {
                "reward": float(index),
                "steps": [{"misc": {"action_misc": {"completion_id": completion_id}}}],
            }
        }
    }


async def varied_token_cost_rollout(task, config) -> dict:
    from tinker_cookbook.completers import TokensWithLogprobs

    from platoon.train.tinker.proxy import proxy_interactions

    collection = await varied_reward_rollout(task, config)
    root = collection["trajectories"]["root"]
    index = int(root["reward"])
    root["reward"] = 1.0
    completion_id = root["steps"][0]["misc"]["action_misc"]["completion_id"]
    root["misc"] = {"model_call_spans": {completion_id: [1, 2]}}
    proxy_interactions.get()[completion_id].action = TokensWithLogprobs(
        tokens=[15] * (index + 2), maybe_logprobs=[-0.25] * (index + 2)
    )
    return collection


async def mixed_success_token_cost_rollout(task, config) -> dict:
    from platoon.train.tinker.proxy import proxy_interactions

    collection = await varied_token_cost_rollout(task, config)
    root = collection["trajectories"]["root"]
    completion_id = root["steps"][0]["misc"]["action_misc"]["completion_id"]
    output_tokens = len(proxy_interactions.get()[completion_id].action.tokens)
    root["reward"] = float(output_tokens >= 4)
    return collection


def token_cost_reward(trajectory):
    from platoon.utils.token_efficiency import trajectory_token_efficiency_metrics

    metrics = trajectory_token_efficiency_metrics(trajectory)
    return trajectory["reward"] - metrics.get("reward/efficiency_penalty", 0), metrics


async def segfault_rollout(_task, _config) -> dict:
    import ctypes

    ctypes.string_at(0)
    return {"trajectories": {}}


async def spin_rollout(_task, _config) -> dict:
    while True:
        pass
