"""Stats logging utilities with WandB support.

Inspired by areal's StatsLogger:
https://github.com/inclusionAI/AReaL/blob/main/areal/utils/stats_logger.py
"""

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""
    mode: str = "online"  # "online", "offline", "disabled"
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    resume_run_id: str | None = None  # WandB run ID to resume


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
    
    def _print_stats(self, step: int, epoch: int | None, stats: dict[str, float]):
        """Print stats to console in a formatted table."""
        header = f"Step {step}"
        if epoch is not None:
            header = f"Epoch {epoch} | {header}"
        
        logger.info(f"\n{'='*60}")
        logger.info(header)
        logger.info(f"{'='*60}")
        
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
        
        logger.info(f"{'='*60}\n")
    
    def log_config(self, config: Any):
        """Log configuration to WandB."""
        if self._wandb_run is not None:
            try:
                import wandb
                config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else {"config": str(config)}
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

