"""One training step's parts, by stage.  The step itself is ``TrainStep``, in :mod:`platoon.train.trainer.loops`
beside the training loop that runs it; it reads top to bottom in seven stages and calls these:

1. collect settled groups into microbatches: what the step tallies of them and tells the dispatcher (``collect``),
   which of them count under a group admission rule (``counting``), the stale filter and its warning
   (``staleness``), the microbatch it fills (``microbatch``) and the copies of a group's datums it trains
   (``datums``);
2. prepare each microbatch to send: the batch transforms (``batch_transforms``), the zero-advantage filter's metrics
   and the microbatch as it is sent (``microbatch``), and the scale of its advantages (``datums``);
3. update the policy once per minibatch (``update``);
4. commit what became of its groups;
5. publish the new policy (``publish``);
6. advance to the next step;
7. log the step's metrics (``metrics``).
"""
