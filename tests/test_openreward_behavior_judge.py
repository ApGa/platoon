from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "plugins/openreward/platoon/openreward/behavior_judge.py"


def _load_module():
    module_name = "openreward_behavior_judge_test_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _action(
    event_id: str,
    response_id: str,
    tool: str,
    arguments: dict,
    timestamp: str,
    *,
    tool_call_id: str | None = None,
    **extra,
) -> dict:
    tool_call_id = tool_call_id or f"call-{event_id}"
    return {
        "kind": "ActionEvent",
        "id": event_id,
        "source": "agent",
        "timestamp": timestamp,
        "llm_response_id": response_id,
        "tool_name": tool,
        "tool_call_id": tool_call_id,
        "tool_call": {
            "id": tool_call_id,
            "name": tool,
            "arguments": json.dumps(arguments),
        },
        **extra,
    }


def _observation(
    event_id: str,
    action_id: str,
    tool: str,
    text: str,
    timestamp: str,
    *,
    is_error: bool = False,
    tool_call_id: str | None = None,
    kind: str = "ObservationEvent",
    **extra,
) -> dict:
    return {
        "kind": kind,
        "id": event_id,
        "source": "environment",
        "timestamp": timestamp,
        "action_id": action_id,
        "tool_call_id": tool_call_id or f"call-{action_id}",
        "tool_name": tool,
        "observation": {
            "kind": "ProgrammaticToolCallingObservation"
            if tool == "programmatic_tool_calling"
            else "MCPToolObservation",
            "tool_name": tool,
            "is_error": is_error,
            "content": [{"type": "text", "text": text}],
            **extra,
        },
    }


def _step(*, actions=None, observations=None, completion_id: str | None = None, misc=None) -> dict:
    step_misc = dict(misc or {})
    if completion_id is not None:
        step_misc["action_misc"] = {"completion_id": completion_id}
    return {
        "action_events": actions,
        "observation_events": observations,
        "misc": step_misc,
    }


def _safe_summary() -> str:
    return (
        "USER_CONTEXT: Continue checking the two catalog results.\n"
        "COMPLETED: Located the relevant records.\n"
        "PENDING: Validate the final record and report it.\n"
        "LEARNED_PATTERNS: Use distinct catalog keys in parallel."
    )


def test_renderer_groups_parallel_calls_by_response_and_distinct_arguments():
    module = _load_module()
    trajectory = {
        "id": "root",
        "steps": [
            _step(
                actions=[
                    _action("a1", "response-1", "call_tool", {"name": "lookup", "id": 1}, "2026-01-01T00:00:01"),
                    _action("a2", "response-1", "call_tool", {"name": "lookup", "id": 2}, "2026-01-01T00:00:01"),
                    _action("a3", "response-1", "call_tool", {"name": "lookup", "id": 2}, "2026-01-01T00:00:01"),
                ],
                observations=[
                    _observation("o1", "a1", "call_tool", "one", "2026-01-01T00:00:02"),
                    _observation("o2", "a2", "call_tool", "two", "2026-01-01T00:00:02"),
                    _observation("o3", "a3", "call_tool", "two again", "2026-01-01T00:00:02"),
                ],
                completion_id="response-1",
            )
        ],
    }

    rendered = module.render_behavior_judge_prompt(
        "Look up two distinct records.",
        trajectory,
        max_prompt_tokens=4096,
    )

    stats = rendered.metadata["stats"]
    assert stats["action_groups"] == 1
    assert stats["actions"] == 3
    assert stats["parallel_action_groups"] == 1
    assert stats["distinct_call_signatures"] == 2
    assert stats["repeated_identical_calls"] == 1
    assert "G0001 response=response-1 actions=3 distinct_calls=2 parallel=yes" in rendered.user_prompt
    assert rendered.user_prompt.count("call_tool(args_sha=") >= 2


def test_hybrid_history_keeps_pre_compaction_delegation_and_error_and_uses_timestamp_tail():
    module = _load_module()
    trajectory = {
        "id": "root",
        "finish_message": "Validated and integrated the final record.",
        "error_message": None,
        "steps": [
            _step(
                actions=[
                    _action(
                        "delegate",
                        "r-delegate",
                        "launch_subagent",
                        {"goal": "Find candidate records"},
                        "2026-01-01T00:00:01",
                        thought="SECRET_PRE_COMPACTION_COT",
                    )
                ],
                observations=[
                    _observation(
                        "delegate-error",
                        "delegate",
                        "launch_subagent",
                        "Budget unavailable; narrow the request.",
                        "2026-01-01T00:00:02",
                        is_error=True,
                        kind="AgentErrorEvent",
                        error_kind="budget_error",
                    )
                ],
                completion_id="r-delegate",
            ),
            _step(
                actions=[
                    _action(
                        "pre",
                        "r-pre",
                        "programmatic_tool_calling",
                        {"code": "print('before')"},
                        "2026-01-01T00:01:00",
                        reasoning_content="SECRET_REASONING",
                    )
                ],
                observations=[
                    _observation(
                        "pre-result",
                        "pre",
                        "programmatic_tool_calling",
                        "before",
                        "2026-01-01T00:01:01",
                    )
                ],
                completion_id="r-pre",
            ),
            # OpenHands can append the synthetic condensation step after this
            # already-started post-condensation model step.
            _step(
                actions=[
                    _action(
                        "post",
                        "r-post",
                        "call_tool",
                        {"name": "catalog.validate", "arguments": {"id": 7}},
                        "2026-01-01T00:03:00",
                    )
                ],
                observations=[_observation("post-result", "post", "call_tool", "valid", "2026-01-01T00:03:01")],
                completion_id="r-post",
            ),
            _step(
                observations=[
                    {
                        "kind": "Condensation",
                        "id": "condense",
                        "timestamp": "2026-01-01T00:02:00",
                        "summary": _safe_summary(),
                    }
                ],
                completion_id="r-condense",
                misc={"condensation_reasoning": "SECRET_CONDENSATION_COT"},
            ),
        ],
    }
    child = {
        "id": "child",
        "parent_info": {"id": "root", "fork_step": 1},
        "task": {"goal": "expanded prompt", "misc": {"openreward_current_agent_task_goal": "Find candidates"}},
        "finish_message": "Returned two candidate IDs.",
        "steps": [
            _step(
                observations=[{"kind": "ObservationEvent", "observation": {"content": [{"text": "CHILD_RAW_SECRET"}]}}]
            )
        ],
    }
    verifier = {
        "id": "verifier",
        "parent_info": {"id": "root", "fork_step": 1},
        "task": {"goal": "VERIFIER_GOAL_SECRET", "misc": {"subagent_reward_verifier_task": True}},
        "steps": [],
    }
    collection = SimpleNamespace(trajectories={"root": trajectory, "child": child, "verifier": verifier})

    rendered = module.render_behavior_judge_prompt(
        "Find and validate a record.",
        trajectory,
        max_prompt_tokens=8192,
        collection=collection,
    )

    prompt = rendered.user_prompt
    assert "Latest safe condensation summary" in prompt
    assert "LEARNED_PATTERNS: Use distinct catalog keys in parallel." in prompt
    assert "G0001 response=r-delegate" in prompt
    assert "ERROR G0001" in prompt
    assert "budget_error" in prompt
    assert "E00004 ACTION" in prompt  # timestamp, not synthetic step order, retains `post`
    # Programmatic code is an observable action (not hidden reasoning) and is
    # retained compactly even before the condensation boundary.
    assert '"code":"print(\'before\')"' in prompt
    assert "Find candidates" in prompt
    assert "Returned two candidate IDs." in prompt
    assert "CHILD_RAW_SECRET" not in prompt
    assert "VERIFIER_GOAL_SECRET" not in prompt
    assert "SECRET_PRE_COMPACTION_COT" not in prompt
    assert "SECRET_REASONING" not in prompt
    assert "SECRET_CONDENSATION_COT" not in prompt
    assert rendered.metadata["stats"]["delegations"] == 1
    assert rendered.metadata["stats"]["typed_errors"] == 1
    assert rendered.metadata["stats"]["policy_descendants"] == 1
    assert rendered.metadata["stats"]["verifier_branches_excluded"] == 1
    assert rendered.metadata["latest_condensation_event_id"] == "condense"


def test_renderer_is_bounded_and_reports_omissions():
    module = _load_module()
    steps = []
    for index in range(80):
        action_id = f"a-{index}"
        steps.append(
            _step(
                actions=[
                    _action(
                        action_id,
                        f"response-{index}",
                        "call_tool",
                        {"name": "catalog.search", "query": "q" * 1200, "index": index},
                        f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}",
                    )
                ],
                observations=[
                    _observation(
                        f"o-{index}",
                        action_id,
                        "call_tool",
                        "result " + "x" * 2400,
                        f"2026-01-01T01:{index // 60:02d}:{index % 60:02d}",
                    )
                ],
                completion_id=f"response-{index}",
            )
        )
    trajectory = {"id": "large", "steps": steps, "finish_message": "done"}

    rendered = module.render_behavior_judge_prompt(
        "Search all records." + " goal" * 1000,
        trajectory,
        max_prompt_tokens=2048,
    )

    assert rendered.metadata["estimated_prompt_tokens"] <= 2048
    assert "OMITTED" in rendered.user_prompt
    omissions = rendered.metadata["omissions"]
    assert omissions["ledger"]["omitted_entries"] > 0
    assert omissions["tail"]["omitted_events"] > 0


@pytest.mark.parametrize(
    "response",
    [
        'prefix\n```json\n{"status":"pass","passed":true,"reason":"ok","violations":[],"evidence":[]}\n```',
        '```json\n{"status":"pass","passed":true,"reason":"ok","violations":[],"evidence":[]}\n```\nsuffix',
        '```json\n{"status":"pass","passed":true,"reason":"ok","violations":[],"evidence":[]}\n```\n```',
        '{"status":"pass","passed":false,"reason":"ok","violations":[],"evidence":[]}',
        '{"status":"pass","passed":true,"reason":"ok","violations":[],"evidence":[],"score":1}',
        '{"status":"pass","passed":true,"reason":"ok","violations":[],"evidence":[]}',
        '{"status":"fail","passed":false,"reason":"bad","violations":[],"evidence":[]}',
    ],
)
def test_strict_parser_rejects_ambiguous_or_nonconforming_responses(response):
    module = _load_module()
    with pytest.raises(ValueError):
        module.parse_behavior_judgment(response)


@pytest.mark.parametrize(
    ("status", "passed", "violations"),
    [
        ("pass", True, []),
        ("fail", False, ["pure_forwarding"]),
        ("insufficient_evidence", None, []),
    ],
)
def test_strict_parser_accepts_each_consistent_status(status, passed, violations):
    module = _load_module()
    parsed = module.parse_behavior_judgment(
        json.dumps(
            {
                "status": status,
                "passed": passed,
                "reason": "Observable evidence supports this verdict.",
                "violations": violations,
                "evidence": ["G0001"],
            }
        )
    )
    assert parsed["status"] == status
    assert parsed["passed"] is passed


def test_strict_parser_accepts_one_unambiguous_json_fence():
    module = _load_module()
    parsed = module.parse_behavior_judgment(
        "```json\n"
        '{"status":"fail","passed":false,"reason":"Forwarded the task.",'
        '"violations":["pure_forwarding"],"evidence":["G0001"]}'
        "\n```"
    )

    assert parsed["status"] == "fail"
    assert parsed["passed"] is False


class _FakePolicyLLM:
    def __init__(
        self,
        content: str | None = None,
        error: Exception | None = None,
        *,
        bytes_per_token: int = 4,
        token_error: Exception | None = None,
    ):
        self.content = content
        self.error = error
        self.bytes_per_token = bytes_per_token
        self.token_error = token_error
        self.model = "openai/Qwen/Qwen3.6-35B-A3B"
        self.usage_id = "platoon-openreward-behavior-judge"
        self.completion_calls = []
        self.token_count_calls = []
        self.close_calls = 0

    @staticmethod
    def _message_text(messages) -> str:
        return "\n".join(
            item.text
            for message in messages
            for item in message.content
            if isinstance(getattr(item, "text", None), str)
        )

    def get_token_count(self, messages):
        self.token_count_calls.append(messages)
        if self.token_error is not None:
            raise self.token_error
        byte_count = len(self._message_text(messages).encode("utf-8"))
        return max(1, math.ceil(byte_count / self.bytes_per_token))

    async def acompletion(self, *, messages, **kwargs):
        self.completion_calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            message=SimpleNamespace(
                content=[SimpleNamespace(text=self.content)],
                reasoning_content="SEPARATE_REASONING_MUST_NOT_BE_PARSED",
            )
        )

    async def aclose(self):
        self.close_calls += 1


def test_async_judge_uses_policy_llm_messages_and_strips_in_band_reasoning():
    module = _load_module()
    raw = json.dumps(
        {
            "status": "fail",
            "passed": False,
            "reason": "The agent only forwarded the child result.",
            "violations": ["pure_forwarding"],
            "evidence": ["lineage child-1", "G0001"],
        }
    )
    llm = _FakePolicyLLM("private deliberation that must disappear</think>\n" + raw)
    judge = module.OpenRewardBehaviorJudge(
        llm=llm,
        max_prompt_tokens=4096,
    )

    result = asyncio.run(judge.judge("Do the work", {"id": "root", "steps": []}))
    asyncio.run(judge.aclose())
    asyncio.run(judge.aclose())

    assert result["status"] == "fail"
    assert result["passed"] is False
    assert result["training_eligible"] is True
    assert result["raw_response"] == raw
    assert "private deliberation" not in result["raw_response"]
    assert "SEPARATE_REASONING" not in result["raw_response"]
    assert result["model"] == llm.model
    assert result["usage_id"] == llm.usage_id
    messages, completion_kwargs = llm.completion_calls[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert all(isinstance(message, module.Message) for message in messages)
    assert completion_kwargs == {"store": False}
    assert result["prompt_metadata"]["prompt_token_count_source"] == "policy_llm_get_token_count"
    assert result["prompt_metadata"]["exact_prompt_tokens"] <= 4096
    # The wrapper must not close the shallow policy copy's shared transport.
    assert llm.close_calls == 0


def test_policy_tokenizer_adaptively_shrinks_initial_byte_estimate():
    module = _load_module()
    llm = _FakePolicyLLM(
        json.dumps(
            {
                "status": "pass",
                "passed": True,
                "reason": "The agent directly performed and checked the work.",
                "violations": [],
                "evidence": ["G0080"],
            }
        ),
        # Twice as many tokens as the renderer's four-byte estimate.
        bytes_per_token=2,
    )
    trajectory = {
        "id": "large",
        "steps": [
            _step(
                actions=[
                    _action(
                        f"a-{index}",
                        f"r-{index}",
                        "call_tool",
                        {"query": "x" * 1000, "index": index},
                        f"2026-01-01T00:00:{index % 60:02d}",
                    )
                ],
                completion_id=f"r-{index}",
            )
            for index in range(80)
        ],
    }
    judge = module.OpenRewardBehaviorJudge(llm=llm, max_prompt_tokens=2048)

    result = asyncio.run(judge.judge("Search all records", trajectory))

    assert result["status"] == "pass"
    assert len(llm.token_count_calls) >= 2
    metadata = result["prompt_metadata"]
    assert metadata["exact_prompt_tokens"] <= 2048
    assert metadata["render_estimate_budget_tokens"] < 2048
    assert llm.get_token_count(llm.completion_calls[0][0]) <= 2048


def test_renderer_falls_back_to_labeled_byte_estimate_when_counter_is_unavailable():
    module = _load_module()
    llm = _FakePolicyLLM(token_error=RuntimeError("tokenizer unavailable"))

    rendered = module.render_behavior_judge_prompt(
        "goal",
        {"id": "root", "steps": []},
        max_prompt_tokens=4096,
        llm=llm,
    )

    assert rendered.metadata["prompt_token_count_source"] == "utf8_bytes_div_4_fallback"
    assert "not tokenizer-exact" in rendered.metadata["prompt_token_estimator"]
    assert len(llm.token_count_calls) == 1


def test_malformed_or_failed_judge_is_fail_closed():
    module = _load_module()
    malformed = _FakePolicyLLM("hidden thought</think>not json")
    failed = _FakePolicyLLM(error=RuntimeError("policy endpoint unavailable"))

    malformed_result = asyncio.run(
        module.OpenRewardBehaviorJudge(llm=malformed).judge(
            "goal",
            {"id": "root", "steps": []},
        )
    )
    failed_result = asyncio.run(
        module.OpenRewardBehaviorJudge(llm=failed).judge(
            "goal",
            {"id": "root", "steps": []},
        )
    )

    for result in (malformed_result, failed_result):
        assert result["status"] == "insufficient_evidence"
        assert result["passed"] is None
        assert result["training_eligible"] is False
        assert result["violations"] == ["behavior_judge_unavailable"]
        assert result["raw_response"] is None


def test_truncated_unmarked_reasoning_is_never_persisted():
    module = _load_module()
    private_reasoning = "PRIVATE DELIBERATION " * 100
    llm = _FakePolicyLLM(private_reasoning)

    result = asyncio.run(
        module.OpenRewardBehaviorJudge(llm=llm).judge(
            "goal",
            {"id": "root", "steps": []},
        )
    )

    assert result["status"] == "insufficient_evidence"
    assert result["raw_response"] is None
    assert private_reasoning.strip() not in result["reason"]


def test_constructor_keeps_exact_injected_policy_and_aclose_does_not_own_it():
    module = _load_module()
    llm = _FakePolicyLLM()
    judge = module.OpenRewardBehaviorJudge(llm=llm)

    assert judge.llm is llm
    asyncio.run(judge.aclose())
    closed_result = asyncio.run(judge.judge("goal", {"id": "root", "steps": []}))
    assert closed_result["training_eligible"] is False
    assert llm.close_calls == 0


def test_wall_clock_timeout_bounds_all_policy_retries():
    module = _load_module()

    class NeverCompletesPolicy(_FakePolicyLLM):
        def __init__(self):
            super().__init__()
            self.cancelled = False

        async def acompletion(self, *, messages, **kwargs):
            self.completion_calls.append((messages, kwargs))
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    llm = NeverCompletesPolicy()
    judge = module.OpenRewardBehaviorJudge(
        llm=llm,
        timeout_seconds=0.01,
    )

    result = asyncio.run(judge.judge("goal", {"id": "root", "steps": []}))

    assert result["status"] == "insufficient_evidence"
    assert result["training_eligible"] is False
    assert "TimeoutError" in result["reason"]
    assert llm.cancelled is True
    assert llm.completion_calls[0][1] == {"store": False}


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"max_prompt_tokens": True}, TypeError),
        ({"max_prompt_tokens": 4096.0}, TypeError),
        ({"max_prompt_tokens": 1000}, ValueError),
        ({"timeout_seconds": True}, TypeError),
        ({"timeout_seconds": float("inf")}, ValueError),
        ({"timeout_seconds": 0}, ValueError),
    ],
)
def test_constructor_rejects_invalid_prompt_budgets(kwargs, error_type):
    module = _load_module()
    with pytest.raises(error_type):
        module.OpenRewardBehaviorJudge(llm=_FakePolicyLLM(), **kwargs)


@pytest.mark.parametrize("llm", [None, object(), SimpleNamespace(acompletion="not callable")])
def test_constructor_requires_policy_acompletion(llm):
    module = _load_module()
    with pytest.raises(TypeError):
        module.OpenRewardBehaviorJudge(llm=llm)
