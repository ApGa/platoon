"""Configuration loading utilities.

Provides utilities for loading experiment configurations from YAML files
and command-line arguments, similar to areal's load_expr_config.
"""

import argparse
import logging
import sys
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_from_dict(cls: Type[T], data: dict) -> T:
    """Create a dataclass instance from a dictionary, handling nested dataclasses."""
    if not is_dataclass(cls):
        return data

    # Resolve postponed annotations (from __future__ import annotations)
    # so nested dataclass fields are correctly recognized.
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    field_values = {}
    for f in fields(cls):
        field_type = type_hints.get(f.name, f.type)
        if f.name in data:
            value = data[f.name]
            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                field_values[f.name] = _dataclass_from_dict(field_type, value)
            else:
                field_values[f.name] = value
        elif f.default is not MISSING:
            field_values[f.name] = f.default
        elif f.default_factory is not MISSING:
            field_values[f.name] = f.default_factory()

    return cls(**field_values)


def load_yaml_config(config_path: str | Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}


def load_config(
    args: list[str] | None = None,
    config_class: Type[T] | None = None,
    default_config_path: str | None = None,
) -> tuple[T | dict, dict]:
    """Load experiment configuration from YAML and CLI arguments.

    Similar to areal's load_expr_config. Supports:
    - Loading from a YAML config file
    - Overriding with CLI arguments
    - Nested configuration with dot notation

    Args:
        args: Command-line arguments (defaults to sys.argv[1:]).
        config_class: Optional dataclass type to parse config into.
        default_config_path: Default path to config file if not specified via CLI.

    Returns:
        Tuple of (parsed config, raw config dict).

    Usage:
        # In your train script:
        config, raw = load_config(sys.argv[1:], MyConfigClass)

        # From command line:
        python train.py --config path/to/config.yaml --train.batch_size 64
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Load experiment configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=default_config_path,
        help="Path to YAML configuration file",
    )

    # Parse known args to get config path, leave rest for overrides
    known_args, remaining = parser.parse_known_args(args)

    # Load base config from YAML
    config_dict: dict = {}
    if known_args.config:
        config_dict = load_yaml_config(known_args.config)
        logger.info(f"Loaded config from: {known_args.config}")

    # Parse remaining args as overrides (support dot notation)
    overrides = _parse_overrides(remaining)

    # Merge overrides into config
    for key, value in overrides.items():
        config_dict = _set_nested(config_dict, key, value)

    if overrides:
        logger.info(f"Applied {len(overrides)} config overrides")

    # Convert to dataclass if specified
    if config_class is not None:
        try:
            config = _dataclass_from_dict(config_class, config_dict)
        except Exception as e:
            logger.error(f"Failed to parse config into {config_class.__name__}: {e}")
            raise
    else:
        config = config_dict

    return config, config_dict


def _parse_overrides(args: list[str]) -> dict[str, Any]:
    """Parse CLI override arguments.

    Supports formats:
    - --key value
    - --key=value
    - --nested.key value (dot notation)
    """
    overrides: dict[str, Any] = {}
    i = 0

    while i < len(args):
        arg = args[i]

        if not arg.startswith("--"):
            i += 1
            continue

        # Remove leading dashes
        key = arg[2:]

        # Check for = syntax
        if "=" in key:
            key, value = key.split("=", 1)
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            value = args[i + 1]
            i += 1
        else:
            # Boolean flag
            value = "true"

        # Parse value type
        overrides[key] = _parse_value(value)
        i += 1

    return overrides


def _parse_value(value: str) -> Any:
    """Parse a string value into its appropriate type."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # None
    if value.lower() in ("none", "null"):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [_parse_value(v.strip()) for v in value.split(",")]

    # String
    return value


def _set_nested(d: dict, key: str, value: Any) -> dict:
    """Set a nested key in a dictionary using dot notation."""
    keys = key.split(".")
    result = d.copy()
    current = result

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        elif not isinstance(current[k], dict):
            current[k] = {}
        else:
            current[k] = current[k].copy()
        current = current[k]

    current[keys[-1]] = value
    return result


def save_config(config: Any, path: str | Path):
    """Save configuration to a YAML file.

    Args:
        config: Configuration to save (dataclass or dict).
        path: Path to save the YAML file.
    """
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to: {path}")
