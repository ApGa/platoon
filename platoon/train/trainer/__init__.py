"""The trainers, and one start of training through them, from opening the training service to the last save.

``api`` is the trainer API (RFC 0021, experimental): the ``Trainer`` protocol ``platoon train`` runs, what it is built
with, and ``TrainerBase``, whose hooks the lifecycle calls; ``run`` the lifecycle every trainer goes through
(``TrainerRun``: one start's phases, in order) and the run's step clock; ``clocks`` each trained policy's own clock
when they step independently; ``hooks`` the step hooks, work at the end of each step of any trainer, and the sequence
that runs them; ``hook_config`` the hooks a config attaches to the rl trainer; ``registry`` the ``trainer:`` key's
check and the rl trainer's options.
``rl`` is the rl trainer (``PlatoonTinkerRLTrainer``), with its ``train()``; ``loops`` its training loop and stopping
the loops (its step is :mod:`platoon.train.step`); ``state`` what its train and evaluation loops share, the policy they
train among it; ``signals`` the signal handlers, the watchdog and the exceptions that end the loops; ``waiting``
awaiting with heartbeats and deadlines.  The policy itself, its training service among it, is
:mod:`platoon.train.policy`.
"""
