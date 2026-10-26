"""Objectives: what a policy's rows train under.

``api`` is what a learner and a step read of an objective (``Objective``): the loss the Tinker service computes, the
loss inputs only the trainer reads, and the training metrics of each microbatch.  ``policy_gradient`` is the objective
RL steps train under (``PolicyGradient``), with its training metrics (``compute_training_metrics``).  A policy's
objective follows its spec's loss (``train.loss_fn``, :mod:`platoon.train.policy.spec`).
"""
