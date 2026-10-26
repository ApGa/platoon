"""One training step: what it reads of each group it dequeues (``collect``), its datums' loss inputs (``datums``), the
microbatch transforms (``batch_transforms``), the update's training metrics (``update``), what follows the update
(``publish``), the staleness warning (``staleness``) and the step's metrics (``metrics``).

The step itself is still one function, the body of the training loop (``_train_loop`` in
:mod:`platoon.train.trainer.loops`): these modules hold what it calls.
"""
