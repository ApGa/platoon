"""What starts a training run's groups (RFC 0019, section 4), and what they settle into.

- ``source``: a start's draws (``RolloutSource``: the per-task statistics, the order and its samplers, the admission
  rules and the dispatcher), which any trainer that rolls out draws through;
- ``setup``: builds and starts a start's dispatcher;
- ``dispatcher``: starts each group when the horizon, the rollout permits and the label limits allow;
- ``permits``: the rollout permits and their budgets;
- ``runner``: runs one group and settles it;
- ``settled``: what a step dequeues of a group, and what it adds to the statistics.
"""
