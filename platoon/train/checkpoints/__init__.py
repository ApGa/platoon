"""What a full checkpoint saves, and how a start resumes from it.

``saving`` is the cadence and the full checkpoint's ``loop_state``; ``resume`` takes up what a checkpoint saved;
``dispatch_state`` is the file of the dispatcher's state; ``kept_groups`` the groups a checkpoint keeps; ``pruning``
deletes the run's older recovery checkpoints (``checkpoint.keep_recovery``).
"""
