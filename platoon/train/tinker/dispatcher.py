"""Moved: platoon 0.5.0 split this module into :mod:`platoon.train.dispatch.dispatcher`,
:mod:`platoon.train.checkpoints.dispatch_state`.  Its names still import from here, and so do the names it imported,
each with a ``PlatoonDeprecationWarning`` that names its new module, until platoon 0.6.0 removes this module."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "BACKLOG_BATCHES": "platoon.train.dispatch.dispatcher",
        "SEQ_COLUMN": "platoon.train.dispatch.dispatcher",
        "POSITION_COLUMN": "platoon.train.dispatch.dispatcher",
        "STATE_FORMAT": "platoon.train.checkpoints.dispatch_state",
        "STATE_FORMATS": "platoon.train.checkpoints.dispatch_state",
        "Key": "platoon.train.dispatch.dispatcher",
        "_NO_LABEL": "platoon.train.dispatch.dispatcher",
        "BLOCKED_REASONS": "platoon.train.dispatch.dispatcher",
        "Permits": "platoon.train.dispatch.dispatcher",
        "RowReader": "platoon.train.dispatch.dispatcher",
        "DrawRecorder": "platoon.train.dispatch.dispatcher",
        "Token": "platoon.train.dispatch.dispatcher",
        "_Ask": "platoon.train.dispatch.dispatcher",
        "ASK": "platoon.train.dispatch.dispatcher",
        "BoundDraw": "platoon.train.dispatch.dispatcher",
        "GroupRunner": "platoon.train.dispatch.dispatcher",
        "Dispatcher": "platoon.train.dispatch.dispatcher",
        "state_name": "platoon.train.checkpoints.dispatch_state",
        "save_state": "platoon.train.checkpoints.dispatch_state",
        "read_state": "platoon.train.checkpoints.dispatch_state",
        "saved_draws": "platoon.train.checkpoints.dispatch_state",
        "saved_fallback_slots": "platoon.train.checkpoints.dispatch_state",
        "saved_token_positions": "platoon.train.checkpoints.dispatch_state",
        "ResumePlan": "platoon.train.checkpoints.dispatch_state",
        "plan_resume": "platoon.train.checkpoints.dispatch_state",
    },
    homes=(
        "platoon.train.dispatch.dispatcher",
        "platoon.train.checkpoints.dispatch_state",
    ),
)
