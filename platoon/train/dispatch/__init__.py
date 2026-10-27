"""What starts a training run's groups (RFC 0019, section 4), and what they settle into.

``source`` holds a start's draws (``RolloutSource``: the per-task statistics, the order and its samplers, the admission
rules and the dispatcher), which any trainer that rolls out draws through; ``dispatcher`` starts each group when the
horizon, the rollout permits and the label limits allow; ``permits`` holds the rollout permits and their budgets;
``runner`` runs one group and settles it; ``settled`` is what a step dequeues of a group, and what it adds to the
statistics; ``setup`` builds and starts a start's dispatcher.
"""
