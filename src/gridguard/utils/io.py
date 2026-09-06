"""I/O helpers for Parquet, JSON, artifacts, and runs."""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd


def save_parquet(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to Parquet file format ensuring parent directories exist."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_parquet(filepath, index=False, engine="pyarrow")


def load_parquet(filepath: str) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    if not os.path.exists(filepath) and not os.path.isabs(filepath):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        alt = os.path.join(base_dir, filepath)
        if os.path.exists(alt):
            filepath = alt
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")
    return pd.read_parquet(filepath, engine="pyarrow")


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save python object as formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """Load python object from JSON file."""
    if not os.path.exists(filepath) and not os.path.isabs(filepath):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        alt = os.path.join(base_dir, filepath)
        if os.path.exists(alt):
            filepath = alt
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def create_run_directory(base_dir: str = "artifacts/runs", run_id: Optional[str] = None) -> str:
    """Create a unique and deterministic run directory for experiment tracking."""
    if run_id is None:
        import datetime
        import uuid
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        run_id = f"run_{timestamp}_{short_id}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_dir
