"""Shared configuration definitions for Platoon.

This module contains configuration classes that are used across different 
parts of the codebase (rollouts, training, inference, etc.).
"""

from dataclasses import dataclass


@dataclass
class RolloutConfig:
    """Configuration for rollout execution.
    
    This configuration is used for running agent rollouts, whether for 
    training, evaluation, or standalone inference.
    """
    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = 'rollout_results'
    verbose: bool = True
    timeout: int | None = None
    return_dict: bool = False

