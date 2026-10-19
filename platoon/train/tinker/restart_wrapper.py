#!/usr/bin/env python3
"""Deprecated: set ``watchdog.max_restarts`` in the training config instead, and ``platoon train`` restarts itself
after any failure, not only the watchdog's (:mod:`platoon.train.supervise`).  Platoon 0.3.0 removes this module; until
then it works as before, and says so with a :class:`~platoon.PlatoonDeprecationWarning`.

Wrapper script that restarts training on watchdog timeout.

This script runs a training command and automatically restarts it if:
1. The process exits with the watchdog exit code (default: 2)
2. The number of restarts hasn't exceeded the maximum

The training script should be designed to resume from the last checkpoint
when restarted (which the PlatoonTinkerRLTrainer already supports).

Every time the training process exits, and before it is started again, the
wrapper kills the rollout workers (``rollout_isolation: process``) whose trainer
is gone.  Workers lead their own sessions, so a trainer that died without
reaping them (a watchdog exit, a crash, SIGKILL) would otherwise leave them
running next to the restarted trainer; trainers record each live worker on disk
for exactly this (:mod:`platoon.train.tinker.workers.registry`).  Workers whose
trainer still runs, such as another run's sharing the same ``$TMPDIR``, are
never touched.  A SIGTERM to the wrapper itself still ends it at once; the
workers then stop on their trainer's death (PDEATHSIG and their own termination
deadline).

Usage:
    python -m platoon.train.tinker.restart_wrapper \\
        --max-restarts 5 \\
        --watchdog-exit-code 2 \\
        -- python -m my_plugin.train_tinker --config config.yaml

Or as a Python API:
    from platoon.train.tinker.restart_wrapper import run_with_restart
    run_with_restart(["python", "-m", "my_plugin.train_tinker"], max_restarts=5)
"""

import argparse
import logging
import signal
import subprocess
import sys
import time
from typing import Sequence

from platoon import deprecations

#: How long the wrapper's sweep gives a dead trainer's workers between SIGTERM and SIGKILL.
WORKER_SWEEP_GRACE_SECONDS = 10.0
#: What replaces this module, and the release that removes it.
DEPRECATION = deprecations.Deprecation(
    path="python -m platoon.train.tinker.restart_wrapper (run_with_restart)",
    replacement="watchdog.max_restarts in the training config",
    removed_in="0.3.0",
)

# Use a dedicated logger that doesn't affect the root logger
# This prevents interference when restart_wrapper is imported by subprocess code
logger = logging.getLogger("platoon.restart_wrapper")


def _configure_restart_wrapper_logging():
    """Configure logging for restart wrapper. Only call when running as main script."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [RESTART-WRAPPER] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def stop_orphaned_workers(
    *,
    registry: str | None = None,
    grace_seconds: float = WORKER_SWEEP_GRACE_SECONDS,
) -> list[int]:
    """Kill the rollout workers whose trainer is gone (the one that just exited, or any other); returns their
    pids; never raises.

    Not only the child's: the trainer may run under a shell or a launcher of its own.  ``registry`` is the
    trainers' worker registry (default: the one under ``$TMPDIR``, where their supervisors keep it).
    """

    try:
        from platoon.train.tinker.workers.registry import registry_dir, sweep_orphaned_workers

        if registry is None:
            from platoon.train.tinker.workers.protocol import ipc_dir

            registry = registry_dir(ipc_dir())
        stopped = sweep_orphaned_workers(registry, grace_seconds=grace_seconds)
    except Exception as e:
        logger.warning(f"Could not check for rollout workers the trainer left running: {e}")
        return []
    if stopped:
        logger.warning(f"Stopped {len(stopped)} rollout worker(s) the trainer left running: {stopped}")
    return stopped


def run_with_restart(
    command: Sequence[str],
    max_restarts: int = 5,
    watchdog_exit_code: int = 2,
    restart_delay_seconds: float = 10.0,
    worker_registry: str | None = None,
    worker_sweep_grace_seconds: float = WORKER_SWEEP_GRACE_SECONDS,
) -> int:
    """Run a command and restart it on watchdog timeout.

    Args:
        command: The command to run (list of strings)
        max_restarts: Maximum number of restarts before giving up
        watchdog_exit_code: The exit code that triggers a restart
        restart_delay_seconds: Delay before restarting
        worker_registry: Where trainers record their rollout workers (default: under ``$TMPDIR``); after every
            exit, before any restart, the workers whose trainer is gone are killed
        worker_sweep_grace_seconds: How long those workers get between SIGTERM and SIGKILL

    Returns:
        The final exit code of the process
    """
    deprecations.warn([DEPRECATION])
    restart_count = 0

    while True:
        logger.info(f"Starting training (attempt {restart_count + 1})")
        logger.info(f"Command: {' '.join(command)}")

        start_time = time.time()
        process = None

        try:
            # Use Popen so we can forward signals to the child
            process = subprocess.Popen(command)
            exit_code = process.wait()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, forwarding to child process...")
            if process is not None:
                # Forward SIGINT to child
                process.send_signal(signal.SIGINT)
                try:
                    # Give it a few seconds to exit gracefully
                    exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.info("Child didn't exit, sending SIGTERM...")
                    process.terminate()
                    try:
                        exit_code = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.info("Child still running, sending SIGKILL...")
                        process.kill()
                        exit_code = process.wait()
                stop_orphaned_workers(registry=worker_registry, grace_seconds=worker_sweep_grace_seconds)
            return 130  # Standard exit code for SIGINT

        # Whatever the exit, nothing the trainer started may run on, least of all next to the next trainer.
        stop_orphaned_workers(registry=worker_registry, grace_seconds=worker_sweep_grace_seconds)
        elapsed = time.time() - start_time

        if exit_code == 0:
            logger.info(f"Training completed successfully (elapsed: {elapsed:.0f}s)")
            return 0

        if exit_code == watchdog_exit_code:
            restart_count += 1

            if restart_count > max_restarts:
                logger.error(
                    f"Training hung and was killed by watchdog. Max restarts ({max_restarts}) exceeded. Giving up."
                )
                return exit_code

            logger.warning(
                f"Training hung and was killed by watchdog (exit code {exit_code}). "
                f"Restarting in {restart_delay_seconds}s... "
                f"(restart {restart_count}/{max_restarts})"
            )
            time.sleep(restart_delay_seconds)
            continue

        # Non-zero, non-watchdog exit code - don't restart
        logger.error(
            f"Training exited with code {exit_code} (elapsed: {elapsed:.0f}s). "
            f"Not restarting (only restart on watchdog exit code {watchdog_exit_code})."
        )
        return exit_code


def main():
    _configure_restart_wrapper_logging()

    parser = argparse.ArgumentParser(
        description="Restart training on watchdog timeout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=5,
        help="Maximum number of restarts (default: 5)",
    )
    parser.add_argument(
        "--watchdog-exit-code",
        type=int,
        default=2,
        help="Exit code that triggers restart (default: 2)",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=10.0,
        help="Delay in seconds before restart (default: 10)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (after --)",
    )

    args = parser.parse_args()

    # Handle the case where command starts with '--'
    command = args.command
    if command and command[0] == "--":
        command = command[1:]

    if not command:
        parser.error("No command specified. Use -- before the command.")

    exit_code = run_with_restart(
        command=command,
        max_restarts=args.max_restarts,
        watchdog_exit_code=args.watchdog_exit_code,
        restart_delay_seconds=args.restart_delay,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
