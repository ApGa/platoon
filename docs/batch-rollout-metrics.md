# Batch rollout metrics

Both the Tinker and AReaL trainers report reward and trajectory statistics for
the groups consumed by the logged training batch under `train/batch/`.
Use these metrics when comparing reward against training progress:

| Metric | Population |
| --- | --- |
| `task_reward/avg`, `/min`, `/max`, `/count` | Returned root rollouts in the batch |
| `task_reward_valid/avg` | Fraction of returned roots with a completed reward |
| `task_reward_at_k_mean/avg`, `task_reward_at_k_min/avg`, `task_reward_at_k_max/avg` | Mean of each group's mean, worst, or best root reward |
| `root_reward/*`, `reward/*` | Available root or trajectory reward components |
| `num_steps`, `num_input_tokens`, `num_output_tokens` | Trajectories, including children |
| `avg_input_tokens_per_step`, `avg_output_tokens_per_step` | Per-trajectory token counts divided by steps (at least one) |
| `groups` | Groups contributing batch statistics, even if they have no trainable datums |

Each observation metric has `/avg`, `/min`, `/max`, and `/count`. Counts follow
that metric's population. For example, eight zero-reward roots and two
one-reward roots give `task_reward/avg = 0.2`; the equally weighted mean of
their two group means is `task_reward_at_k_mean/avg = 0.5`. Missing optional
reward components do not contribute synthetic zeros.

Batch metrics describe outcomes before datum filtering, splitting, or padding. A
group with all zero advantages still contributes its observed rewards even
when its datums are omitted from model computation. Partial root rewards
remain visible, with `task_reward_valid` indicating completeness. Rollouts
with no returned reward do not become zero-reward observations.

The batch boundary is the outer training batch. With multiple minibatches or
microbatches, it covers all groups consumed across them, as do the existing
training-step logs. A batch that skips its optimizer update can still report
outcomes; existing workload and optimizer metrics indicate what was submitted.

## Completion statistics and filtering

Workflow reward and trajectory statistics now use
`<scope>/rollout_completion/`. These describe
available outcomes recorded by rollout producers between tracker exports;
they can include groups prefetched for future batches. Operational workflow
and throughput counters keep their existing names and export-window semantics.
Tinker exports completion distributions with `/avg`, `/min`, `/max`, and
`/count`; AReaL retains its scalar observation means (without `/avg`) so its
rollout controller can combine worker statistics with the existing count weights.

Tinker attaches small CPU observations to its existing task-result object and
aggregates them after the trainer's staleness check. `train/batch/groups_dequeued`,
`train/batch/stale_groups`, and `train/batch/groups_without_stats` expose queue
coverage. The last counter includes failed `None` results and legacy/custom
workflows without this metadata. Empty native task results can still carry
observed rewards. Workload totals continue to include stale inference work.

AReaL reads the metric fields already returned with each accepted group,
before trainer-side transforms. It materializes only those small fields for
statistics; token and routing tensors follow their existing training path.
Legacy distributed trainers combine observations across data-parallel ranks
before logging, preserving global counts without counting model-parallel replicas.
Groups discarded by the workflow or rollout controller before `prepare_batch`
returns are outside the batch population. Their available completion statistics
and existing rejection/workload counters remain separate.

The implementation changes logging only. It does not change rewards,
advantages, loss normalization, filtering decisions, or optimizer updates, and
adds no fields to the Tinker service API. Existing W&B history is unchanged;
the new namespaces apply to runs launched with this code.
