# Judges and verifiers

**Goal.** Grade what an environment cannot grade on its own: a subagent's work, judged by a verifier agent; how a
subagent worked, audited by a behavior judge; whether a parent used a subagent's work, checked by a parent-usage
judge; and model-graded rewards through a judge endpoint. You will write a behavior judge and a parent-usage
judge, attach them to an episode program, and test them on CPU.

**Prerequisites.**

- [Subagents](../tutorials/subagents.md): spawning, `agents.child_grading` and verifier agents.
- [Episode programs](../tutorials/programs.md): judges are attached by a program.

## Which judge does what

| Mechanism | Judges | Turned on by | Effect on training |
|---|---|---|---|
| Verifier agent | a finished subagent's result, by inspecting the environment | `agents.child_grading: verifier` on the environment entry | The subagent's reward is the verdict's score; a verdict that is not training-eligible keeps the subagent out of policy training |
| Behavior judge | how a verified subagent worked, from its trajectory | a program sets `ep.behavior_judge` before it runs the root | Gates the verifier's score: a `fail` makes it 0, and an unusable verdict makes the subagent ineligible |
| Parent-usage judge | whether a parent used a successful subagent's work | a program adds the `apply_subagent_parent_usage_gates` finalizer | A subagent its parent did not use is kept out of policy training; its reward still counts in statistics and baselines |
| Judge endpoint and judge calls | anything your code asks a model | `ep.judge_endpoint()` in a program or finalizer; `ctx.chat(..., kind="judge")` in a harness | None by itself: the calls are recorded under the trajectory's `judge_calls` for audit and never trained, and use no step budget |

Verifiers and behavior judges fail closed: one that raises, returns a malformed verdict or cannot run makes the
judged subagent ineligible for policy training, never a reward of 0. A parent-usage judge that fails changes
nothing, since uncertain evidence never penalizes a verified subagent.

## Verifier agents

With `child_grading: verifier`, every subagent that completes (or runs out of steps) is judged before its parent
sees the result. Platoon runs a `verifier`-role agent with the subagent's harness and options, on a fork of the
subagent's session, with `agents.verifier.access` (default `read_only`) and `agents.verifier.max_steps`
(default 20). Its goal names the subagent's goal, final message and error, tells it not to trust the subagent's
summary, and asks for a JSON object as its final message:

```json
{
  "status": "verified",
  "score": 1.0,
  "summary": "short verdict",
  "passed_claims": ["claim that was verified"],
  "failed_claims": ["claim that failed verification"],
  "evidence": ["tool-backed evidence you inspected"]
}
```

| `status` | `score` must be | Without a `score` |
|---|---|---|
| `verified` | above 0 | 1.0 |
| `partial` | strictly between 0 and 1 | 0.5 |
| `failed`, `insufficient_evidence` | 0 | 0.0 |

A verdict is **training-eligible** when the verifier completed with a final message, its status is one of the four
and its score is consistent with that status. The subagent then trains on the score. Otherwise (no JSON, an unknown
status, an inconsistent score, a verifier that crashed, timed out or could not launch) the subagent keeps a score of
0 in the statistics but none of its tokens train. Verifier trajectories never train.

To make a harness work as a verifier, it only has to end with the JSON object as its final message: a
`platoon-codeact` verifier calls `finish(...)` with it, and a `platoon-react` verifier writes it as its last reply.
A verifier with `read_only` access sees only the tools the environment marks read-only, so declare the tools that
inspect state as read-only ([Write a custom
environment](../tutorials/custom-environment.md#step-5-mark-read-only-tools)).

The judged subagent's trajectory records:

| Where | What |
|---|---|
| `misc["subagent_reward_judgment"]` | The effective verdict, with `training_eligible` |
| `misc["subagent_outcome_judgment"]`, `misc["subagent_behavior_judgment"]` | The verifier's and the behavior judge's own verdicts, when a behavior judge ran |
| the last step's `reward_misc` | `reward/success` and `reward/subagent_judgment` (the score), plus `reward/subagent_outcome_judgment` and `reward/subagent_behavior_gate` with a behavior judge |
| `misc["exclude_from_policy_training"]` | `true` when the verdict is not training-eligible |

Settings: `agents.verifier.enabled`, `max_steps` and `access` on the environment entry, and the rollout config's
`enable_subagent_verifiers`, which switches verifiers off everywhere, for example in evaluation
([Subagents](../tutorials/subagents.md#verifiers-and-judges)).

## Behavior judges

A behavior judge audits how a verified subagent reached its result, beside the verifier: for example, that it
looked at the environment instead of guessing, or that it did not loop. It is any object with an async `judge`
method that takes the subagent's goal and trajectory and returns a verdict:

```python
# my_judges/behavior.py
from typing import Any

import platoon as pt
from platoon.programs import single


class UsedTools:
    """A behavior judge: a subagent passes when it called at least one environment tool."""

    async def judge(self, *, goal: str, trajectory: pt.Trajectory) -> dict[str, Any]:
        calls = trajectory.misc.get("tool_calls", [])
        if not calls:
            return {"status": "fail", "passed": False, "reason": "The subagent answered without calling a tool."}
        return {"status": "pass", "passed": True, "reason": f"The subagent made {len(calls)} tool calls."}


@pt.program("my-audited")
async def audited(ep: pt.EpisodeContext) -> pt.Outcome:
    """The default program, with every verified subagent's behavior audited."""

    ep.behavior_judge = UsedTools()  # before the root runs
    return await single(ep)
```

The verdict must pair its fields exactly:

| `status` | `passed` | Effect |
|---|---|---|
| `pass` | `true` | The verifier's score stands |
| `fail` | `false` | The score becomes 0; the subagent still trains, on 0 |
| `insufficient_evidence` | `null` | The score becomes 0, and the subagent is not training-eligible |

A non-empty `reason` is required; other keys (`evidence`, `violations`) are kept. A verdict that breaks these
rules, or a judge that raises, gives the status `unparseable` or `judge_error` and makes the subagent ineligible.
The judge runs concurrently with the verifier, only under `child_grading: verifier`, and only when verifiers are
enabled. The `trajectory` is the subagent's record so far: its steps with their annotations, `misc["tool_calls"]`
and its final message.

A judge that asks a model takes the program's judge endpoint when it is built, for example
`MyModelJudge(ep.judge_endpoint())`: see [judge endpoints](#judge-endpoints) below.

## Parent-usage judges

A subagent can succeed at work its parent then ignores. A parent-usage judge decides, after the episode, whether
the parent used each successful subagent's contribution; subagents it was not used for keep their reward in the
statistics and baselines but none of their tokens train. It is any object with an async `judge_parent_usage`
method:

```python
# my_judges/usage.py
import functools
from typing import Any

import platoon as pt
from platoon.harness.verify import apply_subagent_parent_usage_gates
from platoon.programs import single


class QuotesTheChild:
    """A parent-usage judge: the parent used a subagent when its final message contains the subagent's answer."""

    async def judge_parent_usage(
        self,
        *,
        parent_goal: str,
        parent_trajectory: pt.Trajectory,
        child_goal: str,
        child_trajectory: pt.Trajectory,
    ) -> dict[str, Any]:
        answer = (child_trajectory.finish_message or "").strip()
        if not answer:
            return {"status": "insufficient_evidence", "used": None, "reason": "The subagent gave no answer."}
        if answer in (parent_trajectory.finish_message or ""):
            return {"status": "used", "used": True, "reason": "The parent's answer quotes the subagent's."}
        return {"status": "not_used", "used": False, "reason": "The parent's answer does not quote the subagent's."}


@pt.program("my-usage-gated")
async def usage_gated(ep: pt.EpisodeContext) -> pt.Outcome:
    """The default program, then drop the policy data of verified subagents whose parent ignored them."""

    outcome = await single(ep)
    ep.add_finalizer(functools.partial(apply_subagent_parent_usage_gates, judge=QuotesTheChild()))
    return outcome
```

| `status` | `used` | Effect |
|---|---|---|
| `used` | `true` | Nothing changes |
| `not_used` | `false` | The subagent's tokens do not train (`exclude_from_policy_training`) |
| `insufficient_evidence` | `null` | Nothing changes: uncertain evidence never penalizes a verified subagent |

The gate audits only subagents that have a training-eligible verifier verdict with a positive score, so it needs
`child_grading: verifier`. It records the verdict in the subagent's `misc["subagent_parent_usage_judgment"]` and
the components `reward/subagent_parent_usage_gate` and `reward/subagent_parent_usage_judgment_valid`. It runs up
to four judges at once (`max_concurrency`).

## Judge endpoints

Model-graded rewards, critiques and model-backed judges call a model through the gateway on a `judge` agent:

| Where | How |
|---|---|
| An episode program or a finalizer | `endpoint = ep.judge_endpoint()`: an OpenAI-compatible `base_url`, `api_key` and `model` on a `judge` agent under the root. Send `endpoint.headers("judge")` with each request |
| A function harness | `await ctx.chat(messages, kind="judge")` |
| A harness with its own client | the header `X-Platoon-Call-Kind: judge` on a request to `ctx.endpoint` |

Judge calls sample **the policy being trained**: the gateway serves only that model. They are recorded under the
`judge_calls` of the calling agent's trajectory (the root's, for `ep.judge_endpoint()`), never trained, and use no
step budget. The policy can learn to game a verdict that it writes itself, so keep such verdicts out of the reward,
or use them only where the policy cannot influence them, as the
[episode-programs tutorial](../tutorials/programs.md#2-write-the-finalizer) does with its critique. A judge model of
your own (another provider's API, a rubric checker) is called from your code directly; platoon does not record
those calls.

## Judges in an evaluation

An evaluation can run without judges: `inference.judges: false` starts no verifier agent or behavior judge (a child
graded by verifier shares the episode's outcome, and its trajectory records `child_grading: {configured: verifier,
applied: none, reason: judges_off}`), skips parent-usage judges, and refuses judge agents and `judge` calls with
`platoon.gateway.JudgesOffError`, which stops the evaluation (exit 78). A program that judges reads
`EpisodeContext.judges` and does without:

```python
@pt.program("acme-critiqued")
async def critiqued(ep: pt.EpisodeContext) -> pt.Outcome:
    if ep.judges:
        ep.add_finalizer(make_critique(ep.judge_endpoint()))
    await ep.run()
    return await ep.grade()
```

An entry that grades children by verifier and has a reward processor or a program of its own is refused while judges
are off, since its code may read the verdicts no verifier gives: such an entry states the fallback itself with
`agents.child_grading: none`. A rollout function of your own (not `harness_rollout`) reads the switch from
`RolloutConfig.judges` and must honour it itself. [Turn judges off](../how-to/evaluate-hosted-models.md#turn-judges-off)
has the details.

## Test judges on CPU

A judge is an object with an async method, so a test builds trajectories by hand and awaits it:

```python
# tests/test_my_judges.py
import asyncio

from my_judges.behavior import UsedTools
from my_judges.usage import QuotesTheChild
from platoon.trajectory import Trajectory


def test_a_subagent_that_called_no_tool_fails_the_audit():
    verdict = asyncio.run(UsedTools().judge(goal="Read key a", trajectory=Trajectory(id="child")))
    assert (verdict["status"], verdict["passed"]) == ("fail", False)


def test_a_parent_that_quotes_the_answer_used_it():
    parent = Trajectory(id="root", finish_message="The total is 42.")
    child = Trajectory(id="child", finish_message="42")
    verdict = asyncio.run(
        QuotesTheChild().judge_parent_usage(
            parent_goal="Add up", parent_trajectory=parent, child_goal="Read a", child_trajectory=child
        )
    )
    assert verdict["used"] is True
```

```bash
PYTHONPATH=. uv run pytest tests/test_my_judges.py -q
```

```text
..                                                                       [100%]
2 passed in ...s
```

`platoon.harness.verify.normalize_behavior_judgment(verdict)` and `normalize_parent_usage_judgment(verdict)` check a
verdict the way platoon will, so a test can assert that it is training-eligible
(`normalized["training_eligible"]`). To see verifiers and judges in a whole episode without a model, run your
program with `platoon.testing.run_episode`, `agents={"child_grading": "verifier"}` and a scripted policy that
answers each agent by its own prompt ([Test an environment, a harness or a program on
CPU](../how-to/test-extensions.md)), or run [`examples/subagents`](../../examples/subagents/) with
`child_grading=verifier`, as its CPU test does.

## Next steps

- [Reward processors and finalizers](rewards.md): change what each trajectory trains on.
- [Credit assignment](../tutorials/credit-assignment.md): how the subagents' rewards become advantages.
- [Extending platoon](index.md): every extension point and how platoon finds it.
