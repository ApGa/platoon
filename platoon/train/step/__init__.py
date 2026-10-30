"""One training step of the rl trainer, by stage.  The step itself is :class:`~platoon.train.step.step.TrainStep`
(``step``), which the training loop (:mod:`platoon.train.trainer.loops`) runs once per step; it reads top to bottom in
seven stages and calls these:

1. collect settled groups into microbatches: what the step tallies of them and tells the dispatcher (``collect``),
   which of them count under a group admission rule (``counting``), the stale filter and its warning
   (``staleness``), the microbatch it fills (``microbatch``) and the copies of a group's datums it trains
   (``datums``);
2. prepare each microbatch to send: the batch transforms (``batch_transforms``), the zero-advantage filter's metrics
   and the microbatch as it is sent (``microbatch``), and the scale of its advantages (``datums``);
3. update the policy once per minibatch, through its learner (:mod:`platoon.train.policy.learner`), under its
   objective (:mod:`platoon.train.objectives`);
4. commit what became of its groups;
5. end the step: the run's step hooks (:mod:`platoon.train.trainer.hooks`) drain and update the policy, the step
   publishes the new policy once (``publish``), and the hooks hear of it;
6. advance to the next step;
7. log the step's metrics (``metrics``).
"""
