"""One start of training, from opening the training service to the last save (``PlatoonTinkerRLTrainer``).

``rl`` is the trainer and its ``train()``; ``loops`` the training loop, its step (``TrainStep``) and stopping the loops;
``state`` what the train and evaluation loops share, the policy they train among it; ``signals`` the signal handlers,
the watchdog and the exceptions that end the loops; ``waiting`` awaiting with heartbeats and deadlines.  The policy
itself, its training service among it, is :mod:`platoon.train.policy`.
"""
