"""Credit: the advantages a group's rollouts train with.

``group`` is a task group's credit, as ``GroupRolloutWorkflow`` assigns it: the group's baselines
(``group_baselines``), broadcast over each rollout's datums, and the error-token filter.  ``platoon.train.workflow``
exports ``group_baselines``.
"""
