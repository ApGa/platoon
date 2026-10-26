"""Moved: platoon 0.5.0 split this module into :mod:`platoon.train.workers.supervisor`,
:mod:`platoon.train.workers.outcomes` and :mod:`platoon.train.workers.fork_servers`.  Its names still import from here,
and so do the names it imported, each with a ``PlatoonDeprecationWarning`` that names its new module, until platoon
0.6.0 removes this module."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "RolloutWorkerSupervisor": "platoon.train.workers.supervisor",
        "signal_live_workers": "platoon.train.workers.supervisor",
        "_LaunchFailed": "platoon.train.workers.supervisor",
        "WorkerOutcome": "platoon.train.workers.outcomes",
        "_remove_worker_files": "platoon.train.workers.outcomes",
        "_remove_event_temporaries": "platoon.train.workers.outcomes",
    },
    homes=(
        "platoon.train.workers.supervisor",
        "platoon.train.workers.outcomes",
        "platoon.train.workers.fork_servers",
    ),
)
