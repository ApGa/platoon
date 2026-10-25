"""One start of training, from opening the training service to the last save (``PlatoonTinkerRLTrainer``).

``rl`` is the trainer and its ``train()``; ``loops`` the training loop, its step (``TrainStep``) and stopping the loops;
``service`` the training service's training client, asked again after each refusal; ``state`` what the train and
evaluation loops share; ``signals`` the signal handlers, the watchdog and the exceptions that end the loops; ``waiting``
awaiting with heartbeats and deadlines.
"""
