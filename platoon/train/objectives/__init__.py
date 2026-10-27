"""Objectives: what a policy's rows train under.

``api`` is what a learner and a step read of an objective (``Objective``): the loss the Tinker service computes, the
loss inputs only the trainer reads, and the training metrics of each microbatch.  ``policy_gradient`` is the objective
RL steps train under (``PolicyGradient``), with its training metrics (``compute_training_metrics``).
``cross_entropy`` is the one supervised steps train under (``CrossEntropy``), and how a step makes its loss the mean
over its trained tokens (``mean_per_token``).  An rl policy's objective follows its spec's loss (``train.loss_fn``,
:mod:`platoon.train.policy.spec`); the sft trainer's is cross-entropy.
"""
