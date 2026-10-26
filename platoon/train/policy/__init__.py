"""Policies: each model that training samples and steps, in the trainer process.

``spec`` is a policy as the config declares it (``PolicySpec``, ``policy_specs``); ``policy`` its state as training
moves it on (``Policy``: its model info, its Tinker clients, its optimizer and its version); ``set`` the run's policies
(``PolicySet``), built where the trainer is; ``service`` its training client, asked of its Tinker service again after
each refusal; ``checkpoints`` the saves of its one training client; ``binding`` what a rollout worker needs to sample
it.  A run trains one policy today, ``main``, which ``train:`` declares.
"""
