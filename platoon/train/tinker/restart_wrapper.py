#!/usr/bin/env python3
"""Wrapper script that restarts training on watchdog timeout.

This script runs a training command and automatically restarts it if:
1. The process exits with the watchdog exit code (default: 2)
2. The number of restarts hasn't exceeded the maximum

The training script should be designed to resume from the last checkpoint
when restarted (which the PlatoonTinkerRLTrainer already supports).

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
import os
import signal
import subprocess
import sys
import time
from typing import Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RESTART-WRAPPER] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_with_restart(
    command: Sequence[str],
    max_restarts: int = 5,
    watchdog_exit_code: int = 2,
    restart_delay_seconds: float = 10.0,
) -> int:
    """Run a command and restart it on watchdog timeout.
    
    Args:
        command: The command to run (list of strings)
        max_restarts: Maximum number of restarts before giving up
        watchdog_exit_code: The exit code that triggers a restart
        restart_delay_seconds: Delay before restarting
        
    Returns:
        The final exit code of the process
    """
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
            return 130  # Standard exit code for SIGINT
        
        elapsed = time.time() - start_time
        
        if exit_code == 0:
            logger.info(f"Training completed successfully (elapsed: {elapsed:.0f}s)")
            return 0
        
        if exit_code == watchdog_exit_code:
            restart_count += 1
            
            if restart_count > max_restarts:
                logger.error(
                    f"Training hung and was killed by watchdog. "
                    f"Max restarts ({max_restarts}) exceeded. Giving up."
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

