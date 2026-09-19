# Critical-path token efficiency

`workflow_config.token_efficiency_reward` attributes cost to each policy
trajectory's subtree. The cost is the largest token-weighted sequential path:
sequential calls add, and overlapping parent/child or sibling calls contribute
the longest branch. This distinguishes sequential subagents inside one PTC
action from `asyncio.gather` in that same action.

Each operation is either a model request or a child call, whose weight is its
own recursively computed critical path. Recorded monotonic start/end times
establish execution order; elapsed seconds never enter the reward. The longest
compatible sequence of operations is computed by weighted interval scheduling.
Verifier branches and unreferenced behavior-judge completions are excluded.
Repeated action references to the same completion are counted once.

With the existing defaults, effective cost is `output_tokens + 0.01 * input_tokens`
on the selected path. The reward subtracts
`min(0.20, 0.05 * log2(1 + effective_cost / 20000))`. Setting the coefficient to
zero retains the cost metrics and leaves rewards unchanged. Group centering
and policy masks retain their existing semantics.

Tinker records model-request intervals in the proxy; episode and subagent
lifecycles record child-call intervals. Historical trajectories and backends
without request timings conservatively charge unplaced model calls sequentially.
Missing timing is exposed as `efficiency/critical_path_missing_spans`; it must
be zero for a fully traced Tinker rollout. Total-tree token metrics remain
available separately from `efficiency/critical_path_*` metrics.
