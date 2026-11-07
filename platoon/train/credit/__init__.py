"""Credit: the advantages a group's rollouts train with.

``group`` is a task group's credit, as ``GroupRolloutWorkflow`` assigns it: the group's baselines
(``group_baselines``), broadcast over each rollout's datums, the error-token filter and what the group trains
(``group_verdict``).  ``platoon.train.workflow`` exports ``group_baselines``.

An environment entry may credit its groups with an estimator instead (``environments[].credit``, RFC 0021): ``api``
says what an estimator is and what it sees of a group, ``runtime`` what the run calls of it (its ``assign`` under a
time limit, its ``settle``, its state in a checkpoint, and the error that stops the run), ``registry`` how a config
names one, and the others hold core's: ``group`` (``credit: group``, today's credit), ``keyed`` (baselines keyed by
task, policy, role or depth), ``reverse_kl`` (on-policy distillation from a frozen teacher), ``running`` (running
statistics across groups, for groups of one rollout) and ``roles`` (an estimator of its own for each agent role, with
a share of the advantage mass per role).
"""
