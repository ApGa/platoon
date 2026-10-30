"""Credit: the advantages a group's rollouts train with.

``group`` is a task group's credit, as ``GroupRolloutWorkflow`` assigns it: the group's baselines
(``group_baselines``), broadcast over each rollout's datums, the error-token filter and what the group trains
(``group_verdict``).  ``platoon.train.workflow`` exports ``group_baselines``.

An environment entry may credit its groups with an estimator instead (``environments[].credit``, RFC 0021): ``api``
says what an estimator is and what it sees of a group, ``runtime`` what the run calls of it (its ``assign`` under a
time limit, its ``settle``, its state in a checkpoint, and the error that stops the run), ``registry`` how a config
names one, and ``group`` and ``keyed`` hold core's two, ``credit: group`` (today's credit) and ``credit: keyed``
(baselines keyed by task, policy, role or depth).
"""
