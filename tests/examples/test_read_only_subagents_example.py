"""The read-only subagents example (``examples/subagents/read-only-subagents.yaml``) trains as its README says, on CPU:
its config validates, and ``platoon train`` runs in process on ``platoon.testing``'s fake Tinker service, its rollouts
sampling a scripted policy through the real gateway on the subagents example's Vault environment.  Each subagent's
tool listing holds ``read`` alone, while the root's holds ``answer`` too; a subagent that calls ``answer`` all the same
gets an error result that says why, the environment never sees the call, its trajectory counts the refusal, and it
reports its sum in text, which the root adds up and answers.  The run trains on to its final checkpoint."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from test_subagents_example import tool_call, vault_policy

from platoon import cli
from platoon.gateway import mcp
from platoon.testing import trainers as checks
from platoon.train.workers import run as worker_run

CONFIG = "examples/subagents/read-only-subagents.yaml"
REFUSAL = "Tool 'answer' is not allowed: agents of role 'subagent' may call only read. The call was not run."
_TOOL_RESPONSES = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)


def answers_anyway(prompt: str) -> str:
    """The subagents example's policy, except that a subagent, which may not answer, calls ``answer`` all the same
    once it has read its keys, and reports their sum in text once that call is refused."""

    responses = _TOOL_RESPONSES.findall(prompt)
    if REFUSAL in responses:
        return f"The sum is {sum(int(value) for value in responses if value.isdigit())}."
    reply = vault_policy(prompt)
    if reply.startswith("The sum is"):  # a subagent that cannot see answer: it calls it anyway
        return tool_call("answer", total=int(reply.removeprefix("The sum is ").rstrip(".")))
    return reply


def test_the_example_config_validates(capsys):
    cli.main(["config", "validate", CONFIG])
    assert capsys.readouterr().out.startswith("ok ")


def test_subagents_see_and_call_read_alone_and_the_run_trains_on(serve_policy, tmp_path, monkeypatch):
    listings: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    list_agent_tools = mcp.list_agent_tools

    async def listed(agent: Any) -> list[Any]:
        tools = await list_agent_tools(agent)
        listings[agent.role].add(tuple(sorted(tool.name for tool in tools)))
        return tools

    collections: list[dict[str, Any]] = []
    convert = worker_run.get_train_data_for_trajectory_collection

    def converted(**kwargs: Any) -> Any:
        collections.append(kwargs["trajectory_collection"])
        return convert(**kwargs)

    monkeypatch.setattr(mcp, "list_agent_tools", listed)
    monkeypatch.setattr(worker_run, "get_train_data_for_trajectory_collection", converted)
    serve_policy(answers_anyway)
    service = checks.FakePolicyService()
    settings = (
        f"log_path={tmp_path}",
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.workflow_config.group_size=2",
        "eval.every=0",
        "watchdog.enabled=false",
    )
    with checks._faked(service), checks._restored_signals():
        cli.main(["train", CONFIG, *settings])

    run = tmp_path / "subagents" / "read-only-subagents"
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [(row["name"], row["batch"]) for row in rows] == [("recovery-000000", 0), ("final", 2)]
    # The root (role policy) lists every tool of the environment; each subagent read alone.
    assert dict(listings) == {"policy": {("answer", "read")}, "subagent": {("read",)}}

    roots, subagents = [], []
    for collection in collections:
        for trajectory in collection["trajectories"].values():
            (subagents if trajectory["misc"]["agent"]["role"] == "subagent" else roots).append(trajectory)
    assert roots and subagents
    for subagent in subagents:
        # Its answer call was refused once and never reached the environment, which ran its reads only.
        assert subagent["misc"]["tools_refused"] == {"count": 1, "first": "answer"}
        assert {call["name"] for call in subagent["misc"].get("tool_calls", [])} <= {"read"}
        assert subagent["finish_message"].startswith("The sum is")
    for root in roots:
        assert "tools_refused" not in root["misc"] and root["finish_message"].startswith("Answered")
