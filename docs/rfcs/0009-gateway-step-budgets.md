# RFC 0009: Step budgets enforced at the gateway

Status: deferred. The harness layer bounds steps in the harness spec and grades context overflow as a policy failure; this RFC covers the harness-agnostic backstop that is still missing.

**Decision:** deferred. Harness specs pass the run's step budget to the harness (`{max_steps}`), and a context
overflow is graded as a policy failure, which covers the harnesses platoon ships. A budget in the gateway has to
define a step when one agent step can make several model calls in parallel. This RFC comes back when a harness
with no step setting of its own needs training.

**Implemented in part:** option A, for specs that opt in. A spec with `[budget] enforce = "gateway"` has its root
agent's `policy` calls capped at the run's step budget (`AgentSession.call_cap`). Past it, the gateway answers with a
final assistant message without tool calls, which is neither sampled nor recorded, and the episode is graded with
`reason="budget"`. The OpenHands CLI spec (`openhands@1.16.0`), which had no step setting, was the first to use it,
until RFC 0016 replaced it with the `openhands` SDK harness, which stops at its own step limit. Agents the harness
mints over HTTP share the root's cap, so the cap bounds their policy calls and the root's together, and a harness in
a task sandbox may make only `policy` calls (its `aux` and `judge` calls are refused with 403), since code the policy
runs there can read the agent's key and would otherwise sample past the cap. The objection below, that a step is not
a call, matters less for a harness that makes one policy call per step; option B remains the general answer.

## Summary

Enforce a per-agent step budget in the model gateway, so that a harness without its own step limit still ends its episode when the policy runs out of budget, and the episode is graded instead of running until the context window or the wall clock stops it.

## Motivation

In test runs of `mini-swe-agent` with Qwen3-4B on toy Harbor tasks, half of the rollouts looped for about 60 turns until the prompt exceeded the 16k context window, the harness crashed, and the rollouts were excluded as infrastructure errors. Two changes address this:

1. **Specs take the run's step budget.** A spec references `{max_steps}`, filled from the task's or run's `max_steps`, else from the spec's own `max_steps` default. `mini-swe-agent` passes it as `agent.step_limit`, and hitting the limit exits 0, so the episode is graded. `platoon-bash` passes it as `--max-turns`.
2. **Context overflow is a policy failure.** The gateway counts the prompts it refuses for exceeding the context window per agent. When the harness then exits with an error, `harness_rollout` grades the session and marks the outcome `reason="context_overflow"`, unless the environment sets `on_context_overflow: exclude`.

Neither helps a harness that has no step-limit setting (Codex, for example) and never overflows within the window: it runs until `timeout`.

## Proposal

Give each agent a budget at the gateway. When the budget is spent, the gateway answers the next model call with a final assistant message that makes no tool calls (for example "You have used your step budget; stop here."), marks the agent `budget_exhausted`, and `harness_rollout` grades the session with `reason="budget"`.

## The hard part: what is a step

A harness step is not a model call:

- **Parallel calls.** A step can issue several model calls at once (parallel sampling, a judge alongside the policy call, harness-native subagents sharing one key).
- **Side calls.** Harnesses make calls that are not policy turns: titles, summaries, compaction, memory consolidation.
- **Retries.** A retried call is the same step.

Counting raw calls would charge a harness for its implementation details, so the budget would mean different things across harnesses and could not be compared across a mixture.

Options:

| Option | How a step is counted | Weakness |
|---|---|---|
| A. Count `main` calls only | gateway call kinds from headers and per-harness role patterns | relies on classifying side calls; parallel main calls still count separately |
| B. Count history growth | a call that extends the agent's conversation by a new assistant turn is a step; parallel calls on the same prefix count once | needs a prefix ledger (token-in/token-out, RFC 0005) |
| C. Count tool executions | steps are tool calls through the MCP tool plane | misses native tools that run inside the sandbox |
| D. Leave budgets to specs | today's behavior | harnesses without a step setting are unbounded |

Recommendation: option B once the token-in/token-out ledger lands (RFC 0005), because the ledger already knows which calls extend which prompt, so parallel calls on the same prefix and retries collapse naturally. Until then, keep D.

## Rollout plan

1. Land the TITO prefix ledger (RFC 0005).
2. Add `steps` to the agent ledger and a `limits.max_steps` per role; answer with a terminal message when it is spent.
3. Map `budget_exhausted` to a graded outcome with `reason="budget"`; log `harness/steps` per agent.

## Open questions

- Should the terminal message be harness-specific (some harnesses retry an empty answer)?
- Should a budget-exhausted child in a recursive tree end only itself or its subtree?
