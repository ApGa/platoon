"""How ``platoon train`` runs training: in its own process, or supervised, restarting it after failures.

With ``watchdog.max_restarts`` at 0 (the default), ``platoon train`` trains in its own process.  Above 0 it runs
each start of training in a child process (:func:`main`: ``python -m platoon.train.supervise <run
directory>/config.resolved.yaml``) and starts it again when it fails (:func:`supervise`):

* **What restarts.**  Any exit but success (0), a configuration error (78) and an interrupt (130 or 143, or a start
  that SIGINT or SIGTERM killed), and any exit once ``platoon train`` itself got SIGINT or SIGTERM.  So a training
  service that stalls, which ends the start with a deadline error and exit 1, restarts as the watchdog's exit and a
  crash do.
* **How many times.**  At most ``max_restarts`` restarts, and none after three starts in a row that saved no new
  checkpoint: the newest resumable row of ``checkpoints.jsonl`` must resume at a later step after a start than before
  it, which the recovery checkpoint every start saves at launch does not.  So a crash that every start meets stops
  after three.  ``platoon train`` then exits with the last start's exit code.
* **Each start** resumes from the newest checkpoint, as any start of a run does.  It asks the training service for
  its training client until the service accepts it, for up to 20 minutes, logging each refusal (the trainer's
  ``_open_training_client``), so no delay is waited between starts: a service may hold the dead start's session, and
  with it the only model it trains at once, until the session's lease expires.
* **Leftovers.**  Before the first start, before each restart and when ``platoon train`` exits, the rollout workers
  whose trainer is gone are killed and the sandboxes labelled with the run are reaped, with a count logged
  (:mod:`platoon.train.reaping`).
* **Signals.**  ``platoon train`` passes the first SIGINT or SIGTERM it gets to the running start as SIGUSR2, which
  the trainer answers by stopping at its next step boundary, as on its own first signal, and never takes for a second
  one: a terminal's Ctrl+C reaches every process of its group and a scheduler signals every process of its job, so the
  start may get the signal itself as well.  A later signal reaches it as SIGUSR1, on which the trainer exits at once,
  as on its own second signal.  No start follows a signal.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from platoon.train.reaping import HeldRun, held_run

logger = logging.getLogger(__name__)

__all__ = [
    "EXIT_CONFIG",
    "FRUITLESS_STARTS",
    "INTERRUPTED",
    "main",
    "run_training",
    "start_command",
    "supervise",
]

#: The exit code of a configuration error (``platoon.cli.EXIT_CONFIG``), which no restart can change.
EXIT_CONFIG = 78
#: The exit codes of a start that an interrupt ended: SIGINT's and SIGTERM's, as a shell reports a process they
#: killed.
INTERRUPTED = (128 + signal.SIGINT, 128 + signal.SIGTERM)
#: Starts in a row that saved no new checkpoint, after which no start follows.
FRUITLESS_STARTS = 3
#: The module whose :func:`main` runs one start of a supervised run.
_START_MODULE = "platoon.train.supervise"
#: The signals a supervised start gets from ``platoon train``: stop at the next step boundary, and stop at once.
STOP_SIGNAL = signal.SIGUSR2
STOP_NOW_SIGNAL = signal.SIGUSR1


def run_training(config_path: str | os.PathLike[str], run_dir: str | os.PathLike[str], config: Any) -> int:
    """Train on the run's config snapshot ``config_path`` (``config``, loaded): in this process, or with
    ``watchdog.max_restarts`` above 0 in supervised child processes.  Returns the exit code.

    Either way the run directory records this process (``process.json``), and the run's leftovers are reaped
    before training starts and when it exits (:func:`platoon.train.reaping.held_run`).
    """

    max_restarts = getattr(config.watchdog, "max_restarts", 0)
    with held_run(run_dir, "train", config.environments, workers=True, reap_first=True) as run:
        if max_restarts > 0:
            return supervise(
                start_command(config_path),
                run=run,
                max_restarts=max_restarts,
                watchdog_exit_code=config.watchdog.exit_code if config.watchdog.enabled else None,
            )
        from platoon.train.tinker import train as training  # the Tinker stack: imported only to train

        training.run_tinker_training(["--config", str(config_path)])
    return 0


def start_command(config_path: str | os.PathLike[str]) -> list[str]:
    """The command of one start of a supervised run on the config snapshot ``config_path``."""

    return [sys.executable, "-m", _START_MODULE, str(config_path)]


def supervise(
    command: Sequence[str],
    *,
    run: HeldRun,
    max_restarts: int,
    watchdog_exit_code: int | None = None,
) -> int:
    """Run ``command``, one start of training, again after each failure, as the module docstring says; returns the
    exit code of the last start (``128 + signal`` for a start a signal killed).

    ``run`` is the run directory this process holds: the training process is recorded in it, and its reaper stops
    the run's leftovers before each restart.  ``watchdog_exit_code`` names the watchdog's exit in the log.
    """

    interrupts = _Interrupts()
    previous = {sig: signal.signal(sig, interrupts) for sig in (signal.SIGINT, signal.SIGTERM)}
    starts = fruitless = 0
    try:
        while True:
            if interrupts.signals:
                return 128 + interrupts.signals[0]
            resumes_at = resume_step(run.run_dir)
            starts += 1
            logger.info("Starting training at step %d (start %d of at most %d)", resumes_at, starts, max_restarts + 1)
            code = _run_start(command, run, interrupts)
            run.record_trainer(None)
            if interrupts.signals:
                logger.info("Training stopped (exit code %d) after platoon train was interrupted", code)
                # A start that the signal passed on killed before its trainer took it was interrupted too.
                return 128 + interrupts.signals[0] if code - 128 in (STOP_SIGNAL, STOP_NOW_SIGNAL) else code
            if code == 0:
                logger.info("Training ended (exit code 0)")
                return 0
            if code == EXIT_CONFIG:
                logger.error("Training exited with a configuration error (exit code 78); a restart cannot fix it")
                return code
            if code in INTERRUPTED:
                logger.info("Training was interrupted (exit code %d); not restarting", code)
                return code
            fruitless = fruitless + 1 if resume_step(run.run_dir) <= resumes_at else 0
            why = _exit_reason(code, watchdog_exit_code)
            if fruitless >= FRUITLESS_STARTS:
                logger.error(
                    "Training exited %s, and %d starts in a row saved no new checkpoint: not restarting. Its log "
                    "says why; the run resumes from its last checkpoint when platoon train runs again",
                    why,
                    fruitless,
                )
                return code
            if starts > max_restarts:
                logger.error(
                    "Training exited %s after %d restart(s), which is watchdog.max_restarts: not restarting",
                    why,
                    max_restarts,
                )
                return code
            logger.warning("Training exited %s; restarting it (restart %d of at most %d)", why, starts, max_restarts)
            run.reaper.reap(f"before restart {starts}", workers=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class _Interrupts:
    """The SIGINT and SIGTERM signals ``platoon train`` got (a signal handler), passed on to the running start: the
    first as :data:`STOP_SIGNAL`, later ones as :data:`STOP_NOW_SIGNAL`."""

    def __init__(self) -> None:
        self.signals: list[int] = []
        self.start: subprocess.Popen[bytes] | None = None

    def __call__(self, signum: int, frame: Any) -> None:
        self.signals.append(signum)
        self.pass_on()

    def pass_on(self) -> None:
        """Pass the signals got so far on to the running start, if any."""

        start = self.start
        if start is not None and self.signals:
            with contextlib.suppress(OSError):  # it exited meanwhile
                start.send_signal(STOP_SIGNAL if len(self.signals) == 1 else STOP_NOW_SIGNAL)


def _run_start(command: Sequence[str], run: HeldRun, interrupts: _Interrupts) -> int:
    """Run one start to its end; its exit code, ``128 + signal`` for one that a signal killed."""

    start = subprocess.Popen(list(command))
    interrupts.start = start
    run.record_trainer(start.pid)
    if interrupts.signals:  # one came before the start could be told
        interrupts.pass_on()
    try:
        code = start.wait()
    finally:
        interrupts.start = None
    return 128 - code if code < 0 else code


def resume_step(run_dir: str | os.PathLike[str]) -> int:
    """The step the run resumes at: the ``batch`` of the last row of ``checkpoints.jsonl`` that has a ``state_path``,
    as a resume reads it; 0 for none, where a run starts."""

    try:
        lines = (Path(run_dir) / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("state_path"):
            batch = row.get("batch")
            return batch if isinstance(batch, int) and not isinstance(batch, bool) else 0
    return 0


def _exit_reason(code: int, watchdog_exit_code: int | None) -> str:
    if watchdog_exit_code is not None and code == watchdog_exit_code:
        return f"with code {code}, the watchdog's (no progress for watchdog.timeout_seconds)"
    if code > 128:
        try:
            killer = signal.Signals(code - 128).name
        except ValueError:  # a code above 128 that names no signal Python knows (32, 33, most real-time ones)
            pass
        else:
            return f"with code {code} ({killer} killed it)"
    return f"with code {code}"


# --------------------------------------------------------------------------------------------- one start


def main(argv: Sequence[str] | None = None) -> int:
    """One start of a supervised run: train on the config file ``argv[0]``, the run's ``config.resolved.yaml``.

    Returns ``platoon train``'s exit codes: 0, 78 for a configuration error, 130 for a stop asked before training
    began; any other failure raises, which Python reports with exit code 1.
    """

    # First of all: until the trainer installs its own handlers, the signals platoon train passes on stop the start
    # here, where nothing is left to save, rather than kill it unlogged (their default).
    for sig in (STOP_SIGNAL, STOP_NOW_SIGNAL):
        signal.signal(sig, _stop_before_training)
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(f"usage: python -m {_START_MODULE} CONFIG (the run's config.resolved.yaml)", file=sys.stderr)
        return 64
    logging.basicConfig(  # as platoon's command line does
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from platoon.deprecations import PlatoonDeprecationWarning
    from platoon.prepare import NotPrepared, prepare_command
    from platoon.utils.config import ConfigError

    try:
        from platoon.train.tinker import train as training  # the Tinker stack

        training.run_tinker_training(["--config", args[0]])
    except NotPrepared as exc:  # with the platoon prepare command for the snapshot, which holds the run's overrides
        print(f"platoon: {NotPrepared(exc.problems, command=exc.command or prepare_command(args[0]))}", file=sys.stderr)
        return EXIT_CONFIG
    except (ConfigError, PlatoonDeprecationWarning) as exc:
        print(f"platoon: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        logger.info("Stopped before training began")
        return 128 + signal.SIGINT
    return 0


def _stop_before_training(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())
