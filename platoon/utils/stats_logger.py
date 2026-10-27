"""Stats logging utilities with WandB support.

Inspired by areal's StatsLogger:
https://github.com/inclusionAI/AReaL/blob/main/areal/utils/stats_logger.py
"""

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""

    # "online", "offline" or "disabled".
    mode: str = "online"
    # W&B project (None: ``stats.experiment_name``).
    project: str | None = None
    # W&B team or user that owns the run.
    entity: str | None = None
    # Display name of the run (None: ``stats.trial_name``).
    name: str | None = None
    # Group that collects related runs (None: ``<experiment_name>_<trial_name>``).
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    # API key (None: ``WANDB_API_KEY``).
    api_key: str | None = None
    # URL of a self-hosted W&B server.
    base_url: str | None = None
    # W&B run id to resume.
    resume_run_id: str | None = None


@dataclass
class StatsLoggerConfig:
    """Configuration for the stats logger."""

    experiment_name: str = "experiment"
    trial_name: str = "trial"
    log_dir: str = "logs"
    wandb: WandBConfig = field(default_factory=WandBConfig)
    print_stats: bool = True
    log_interval: int = 1  # Log every N steps


class StatsLogger:
    """Logs training statistics to console and WandB.

    Usage:
        config = StatsLoggerConfig(
            experiment_name="my_experiment",
            trial_name="run_1",
            wandb=WandBConfig(project="my_project")
        )

        stats_logger = StatsLogger(config, exp_config=full_config)

        # During training
        stats_logger.log(step=100, stats={"loss": 0.5, "reward": 0.8})

        # At end
        stats_logger.close()
    """

    def __init__(
        self,
        config: StatsLoggerConfig,
        exp_config: Any = None,
    ):
        self.config = config
        self.exp_config = exp_config
        self.start_time = time.perf_counter()
        self._last_log_step = -1
        self._wandb_run = None
        # The axes and the metric prefixes whose axis W&B was told (log_row).
        self._defined: set[str] = set()

        self._init_logging()

    def _init_logging(self):
        """Initialize logging backends."""
        # Create log directory
        log_path = self._get_log_path()
        os.makedirs(log_path, exist_ok=True)

        # Initialize WandB if enabled
        if self.config.wandb.mode != "disabled":
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            return

        # Set API key if provided
        if self.config.wandb.api_key:
            os.environ["WANDB_API_KEY"] = self.config.wandb.api_key
        if self.config.wandb.base_url:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.base_url

        # Prepare config dict for wandb
        config_dict = {}
        if self.exp_config is not None:
            try:
                config_dict = asdict(self.exp_config)
            except (TypeError, ValueError):
                # exp_config might not be a dataclass
                config_dict = {"exp_config": str(self.exp_config)}

        # Initialize wandb with resume support
        # If resume_run_id is provided, we'll resume that run
        resume_id = self.config.wandb.resume_run_id
        resume_mode = "must" if resume_id else "allow"

        try:
            self._wandb_run = wandb.init(
                id=resume_id,  # Use provided ID or let wandb generate one
                mode=self.config.wandb.mode,
                project=self.config.wandb.project or self.config.experiment_name,
                entity=self.config.wandb.entity,
                name=self.config.wandb.name or self.config.trial_name,
                group=self.config.wandb.group or f"{self.config.experiment_name}_{self.config.trial_name}",
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                config=config_dict,
                dir=self._get_log_path(),
                resume=resume_mode,
            )
            logger.info(f"WandB initialized (id={wandb.run.id}): {wandb.run.url if wandb.run else 'offline'}")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self._wandb_run = None

    def _get_log_path(self) -> str:
        """Get the log directory path."""
        return os.path.join(
            self.config.log_dir,
            self.config.experiment_name,
            self.config.trial_name,
        )

    @property
    def wandb_run_id(self) -> str | None:
        """Get the current WandB run ID for checkpointing."""
        if self._wandb_run is not None:
            try:
                import wandb

                return wandb.run.id if wandb.run else None
            except Exception:
                return None
        return None

    def log(
        self,
        step: int,
        stats: dict[str, float],
        epoch: int | None = None,
        prefix: str = "",
        force: bool = False,
    ):
        """Log statistics for a training step.

        Args:
            step: Current training step.
            stats: Dictionary of metric name -> value.
            epoch: Optional epoch number.
            prefix: Optional prefix for metric names.
            force: If True, skip the log_interval check and always log.
        """
        # Check log interval (skip if force=True)
        if not force and step - self._last_log_step < self.config.log_interval:
            return
        self._last_log_step = step

        # Add prefix to stats
        if prefix:
            stats = {f"{prefix}/{k}": v for k, v in stats.items()}

        # Add step timing
        elapsed = time.perf_counter() - self.start_time
        stats["time/elapsed_seconds"] = elapsed
        stats["time/step"] = step
        if epoch is not None:
            stats["time/epoch"] = epoch

        # Print to console
        if self.config.print_stats:
            self._print_stats(step, epoch, stats)

        # Log to WandB
        if self._wandb_run is not None:
            try:
                import wandb

                wandb.log(stats, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")

    def log_row(self, stats: dict[str, float], *, axis: str) -> None:
        """Log one row without a W&B step of its own, as the rows of policies that step each on its own clock are
        logged: W&B numbers the rows, and each metric plots against the row's ``axis``, one of its metrics (such as
        ``policies/worker/step``), which W&B is told once per metric prefix (``define_metric``): the two-level prefix
        of a ``policies/<id>/...`` name, else the first level.  A row on the run's axis, not a policy's own
        (``policies/<id>/...``), adds ``time/elapsed_seconds``, as :meth:`log` does.  No log interval applies."""

        stats = dict(stats)
        if not axis.startswith("policies/"):
            stats["time/elapsed_seconds"] = time.perf_counter() - self.start_time
        if self.config.print_stats:
            self._print_stats(int(stats.get(axis, 0)), None, stats)
        if self._wandb_run is None:
            return
        try:
            import wandb

            if axis not in self._defined:
                wandb.define_metric(axis)
                self._defined.add(axis)
            for prefix in sorted({_metric_prefix(key) for key in stats} - self._defined):
                wandb.define_metric(f"{prefix}/*", step_metric=axis)
                self._defined.add(prefix)
            wandb.log(stats)
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    def _print_stats(self, step: int, epoch: int | None, stats: dict[str, float]):
        """Print stats to console in a formatted table."""
        header = f"Step {step}"
        if epoch is not None:
            header = f"Epoch {epoch} | {header}"

        logger.info(f"\n{'=' * 60}")
        logger.info(header)
        logger.info(f"{'=' * 60}")

        # Group stats by prefix
        grouped: dict[str, dict[str, float]] = {}
        for key, value in sorted(stats.items()):
            parts = key.split("/", 1)
            if len(parts) == 2:
                group, name = parts
            else:
                group, name = "", key

            if group not in grouped:
                grouped[group] = {}
            grouped[group][name] = value

        # Print grouped stats
        for group in sorted(grouped.keys()):
            if group:
                logger.info(f"\n[{group}]")
            for name, value in sorted(grouped[group].items()):
                if isinstance(value, float):
                    logger.info(f"  {name}: {value:.4f}")
                else:
                    logger.info(f"  {name}: {value}")

        logger.info(f"{'=' * 60}\n")

    def log_config(self, config: Any):
        """Log configuration to WandB."""
        if self._wandb_run is not None:
            try:
                import wandb

                config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else {"config": str(config)}
                wandb.config.update(config_dict)
            except Exception as e:
                logger.warning(f"Failed to log config to WandB: {e}")

    def close(self):
        """Close the logger and finalize logging."""
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"Training complete. Total time: {elapsed:.2f}s")

        if self._wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to close WandB: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _metric_prefix(key: str) -> str:
    """The prefix whose metrics share an axis (:meth:`StatsLogger.log_row`): ``policies/<id>`` for a policy's own
    metrics, else the name's first level."""

    parts = key.split("/")
    return "/".join(parts[:2]) if parts[0] == "policies" and len(parts) > 2 else parts[0]
