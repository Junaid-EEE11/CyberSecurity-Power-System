"""Configuration loading and merging utilities."""

import os
from typing import Any, Dict, Optional
import yaml


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict2 into dict1."""
    result = dict1.copy()
    for k, v in dict2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config file and recursively merge included configs if specified.

    Args:
        config_path: Path to YAML configuration file.
        overrides: Optional key-value dictionary to override parameters.

    Returns:
        Resolved dictionary of configurations.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Check for includes / base configs
    if "includes" in cfg and isinstance(cfg["includes"], list):
        base_dir = os.path.dirname(config_path)
        merged_cfg = {}
        for inc in cfg["includes"]:
            inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            sub_cfg = load_config(inc_path)
            merged_cfg = _deep_merge(merged_cfg, sub_cfg)
        cfg = _deep_merge(merged_cfg, cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save resolved configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
