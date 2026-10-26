"""What starts a training run's groups (RFC 0019, section 4), and what they settle into.

``dispatcher`` starts each group when the horizon, the rollout permits and the label limits allow; ``permits`` holds the
rollout permits and their budgets; ``runner`` runs one group and settles it; ``settled`` is what a step dequeues of a
group; ``setup`` builds and starts a start's dispatcher.
"""
