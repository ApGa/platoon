"""``filter_errors`` marks the tokens of a CodeAct cell that raised, on a rollout through the real gateway.

CodeAct annotates each cell's ``output`` and ``error`` on the model call that wrote it (a cell that raised shows
IPython's traceback as its output); the gateway keeps the annotations in the step's misc, where the error detector
reads them (``platoon.utils.trajectory_error_filtering``).
"""

from __future__ import annotations

import codeact_golden as golden
import pytest
from inline_runs import run_tree

from platoon.envs.service import get_service
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY, detected_error_completion_ids

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
OPTIONS = {
    "prompt_builder": "platoon.envs.number_search.prompts:NumberSearchPromptBuilder",
    "include_reasoning": False,
    "subagents": False,
}


@pytest.mark.asyncio
async def test_the_tokens_of_a_cell_that_raised_are_marked_as_errors(make_gateway):
    from platoon.envs.number_search.env import task_specs

    # The first shell of a process becomes IPython's global one, whose tracebacks are worded differently: keep this
    # rollout's shell out of that slot, as the golden replays do, so neither changes what the other's cells print.
    golden.occupy_global_ipython_shell()
    service = get_service(NUMBER_SEARCH)
    task = await service.get_task("val:2")
    session = await service.open(task, rollout_id="test")
    target = task_specs("val")[2]["misc"]["target"]

    def policy(prompt: str) -> str:
        return (
            '<python>int("many")</python>' if "[Cell 0 Output]" not in prompt else f"<python>guess({target})</python>"
        )

    gateway, _ = make_gateway(policy)
    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=5, options=OPTIONS)
    assert run.result.status == "completed" and run.outcome.reward == 1.0

    collection = run.collection.to_dict()
    steps = next(iter(collection["trajectories"].values()))["steps"]
    failed, _ = (step["misc"]["action_misc"]["completion_id"] for step in steps)
    first, second = (step["misc"] for step in steps)
    assert "ValueError" in first["output"] + first["error"]  # IPython shows the traceback as the output
    assert "Traceback" not in second["output"] and not second["error"]
    assert detected_error_completion_ids(steps) == {failed}

    data = get_train_data_for_trajectory_collection(
        collection, run.interactions, task.id, checkpoint_version=0, filter_errors=True
    )
    marked: list[int] = []
    for datum in data.datums:
        error_mask = datum.loss_fn_inputs[ERROR_ACTION_MASK_KEY].to_torch().bool()
        marked += datum.loss_fn_inputs["target_tokens"].to_torch()[error_mask].tolist()
    assert marked == list(run.interactions[failed].action.tokens)  # the cell that raised, and nothing else
