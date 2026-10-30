"""How a start of training stops: the signal handlers (a first signal stops at the next step boundary, a second
forces the exit), the watchdog that ends a hung process, and the exceptions that end the loops.

``_setup_signal_handlers`` takes the trainer first and is a method of every trainer built on
:class:`~platoon.train.trainer.api.TrainerBase`, which the run's lifecycle calls (:mod:`platoon.train.trainer.run`): to
replace it, replace the method, on the class or an instance, not the function.
"""

import asyncio
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platoon.train.trainer.api import TrainerBase

logger = logging.getLogger(__name__)


# Global flag to track if we've already received a shutdown signal
_SHUTDOWN_REQUESTED = False


class Watchdog:
    """Background thread that monitors for hangs and forcibly exits if no activity.
     If Tinker hangs, the only recovery is to forcibly exit the process.

    Usage:
        watchdog = Watchdog(timeout_seconds=1800)  # 30 min timeout
        watchdog.start()

        # In your main loop:
        watchdog.heartbeat()  # Call periodically to reset the timer

        # When done:
        watchdog.stop()
    """

    def __init__(
        self,
        timeout_seconds: float = 1800,  # 30 minutes default
        exit_code: int = 2,
        on_timeout_callback: Callable[[], None] | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.exit_code = exit_code
        self.on_timeout_callback = on_timeout_callback
        self._last_heartbeat = time.time()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Whether it watches: started, and not stopped since."""

        return self._running

    def heartbeat(self):
        """Call this periodically to indicate the process is making progress."""
        with self._lock:
            self._last_heartbeat = time.time()

    def start(self):
        """Start the watchdog monitoring thread."""
        if self._running:
            return

        self._running = True
        self._last_heartbeat = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started with {self.timeout_seconds}s timeout")

    def stop(self):
        """Stop the watchdog thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Watchdog stopped")

    def _monitor_loop(self):
        """Background thread that checks for hangs."""
        check_interval = min(60.0, self.timeout_seconds / 10)  # Check every minute or less

        while self._running:
            time.sleep(check_interval)

            with self._lock:
                elapsed = time.time() - self._last_heartbeat

            if elapsed > self.timeout_seconds:
                logger.error(
                    f"WATCHDOG: No heartbeat for {elapsed:.0f}s (timeout={self.timeout_seconds}s). "
                    f"Process appears hung. Forcing exit with code {self.exit_code}."
                )

                # Call optional callback (e.g., to save state)
                if self.on_timeout_callback:
                    try:
                        self.on_timeout_callback()
                    except Exception as e:
                        logger.error(f"Watchdog callback failed: {e}")

                # Force immediate exit - this bypasses all cleanup, so tell the rollout workers to stop first
                _stop_rollout_workers()
                os._exit(self.exit_code)

            elif elapsed > self.timeout_seconds * 0.75:
                # Warn when approaching timeout
                logger.warning(
                    f"WATCHDOG: No heartbeat for {elapsed:.0f}s (timeout in {self.timeout_seconds - elapsed:.0f}s)"
                )


def _stop_rollout_workers() -> None:
    """SIGTERM this trainer's live rollout workers before an exit that runs no cleanup; never raises.

    Workers lead their own sessions, so nothing else in the trainer's exit reaches them but PDEATHSIG; on SIGTERM
    each deletes its env sessions and exits within its termination deadline, and ``platoon train`` kills what is
    left before it starts training again (:mod:`platoon.train.reaping`).
    """

    try:
        from platoon.train.workers.supervisor import signal_live_workers

        stopped = signal_live_workers()
    except Exception:
        return
    if stopped:
        logger.warning(f"Sent SIGTERM to {stopped} rollout worker(s) before exiting")


class TerminateTrainLoop(Exception):
    """Exception raised to terminate the train loop."""


class TerminateEvalLoop(Exception):
    """Exception raised to terminate the eval loop for a single evaluation run."""


class TrainInterrupted(Exception):
    """A shutdown signal stopped the train loop where the last completed step is a consistent checkpoint."""


def _setup_signal_handlers(trainer: "TrainerBase", shutdown_event: asyncio.Event) -> None:
    """Set up signal handlers for graceful shutdown.

    First Ctrl+C (or SIGTERM): Sets shutdown_event. The train loop stops at the next step boundary (a batch
    that has applied an update finishes first) and the run saves ``<step>_interrupted`` (``TrainerRun``'s save at
    exit).
    Second Ctrl+C: Forces immediate exit.

    SIGUSR2 and SIGUSR1 are what ``platoon train`` passes on to a start it supervises
    (:mod:`platoon.train.supervise`): SIGUSR2 asks for the stop a first signal makes and never counts as one, so a
    start that got the signal itself as well does not take the two for a second signal; SIGUSR1 forces the exit
    a second signal forces.
    """
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = False

    def force_exit():
        _stop_rollout_workers()
        # Force cleanup of wandb
        try:
            import wandb

            wandb.finish(quiet=True)
        except Exception:
            pass
        os._exit(1)

    def signal_handler(signum, frame):
        global _SHUTDOWN_REQUESTED
        sig_name = signal.Signals(signum).name

        if _SHUTDOWN_REQUESTED:
            # Second signal - force exit
            logger.warning(f"Received {sig_name} again. Forcing immediate exit.")
            force_exit()
        else:
            # First signal - request graceful shutdown
            _SHUTDOWN_REQUESTED = True
            logger.info(f"Received {sig_name}. Requesting graceful shutdown... (press Ctrl+C again to force exit)")
            shutdown_event.set()

    def supervisor_stop_handler(signum, frame):
        if not shutdown_event.is_set():
            logger.info("platoon train was interrupted: stopping at the next step boundary")
            shutdown_event.set()

    def supervisor_force_handler(signum, frame):
        logger.warning("platoon train was interrupted again. Forcing immediate exit.")
        force_exit()

    # Register handlers for SIGINT (Ctrl+C) and SIGTERM, and for what platoon train passes on of them
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR2, supervisor_stop_handler)
    signal.signal(signal.SIGUSR1, supervisor_force_handler)
