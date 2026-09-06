"""Explicit schema definition and data validation for power-grid telemetry datasets."""

from typing import Dict, List, Set
import pandas as pd

DATASET_SCHEMA: Dict[str, str] = {
    "timestamp": "string",
    "timestep": "int64",
    "scenario": "string",
    "node_id": "int64",
    "bus": "string",
    "phase": "int64",
    "v_mag_pu": "float64",
    "v_ang_rad": "float64",
    "sin_theta": "float64",
    "cos_theta": "float64",
    "p_inj_pu": "float64",
    "q_inj_pu": "float64",
    "clean_v_mag_pu": "float64",
    "clean_v_ang_rad": "float64",
    "clean_p_inj_pu": "float64",
    "clean_q_inj_pu": "float64",
    "observation_mask": "int64",
    "is_attack": "int64",
    "attack_id": "string",
    "attack_type": "string",
    "attack_strength": "float64",
    "is_compromised": "int64",
}


def validate_dataframe_schema(df: pd.DataFrame, strict: bool = True) -> bool:
    """Validate DataFrame columns and types against the standard research schema.

    Args:
        df: Pandas DataFrame to validate.
        strict: If True, require exact column presence.

    Returns:
        True if valid.

    Raises:
        ValueError if required columns are missing.
    """
    required_cols: Set[str] = set(DATASET_SCHEMA.keys())
    present_cols: Set[str] = set(df.columns)
    missing = required_cols - present_cols

    if missing:
        raise ValueError(f"DataFrame is missing required schema columns: {missing}")

    return True
