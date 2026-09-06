"""Strict chronological data partitioning without temporal overlap or data leakage."""

from typing import Dict, Tuple
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_days: int,
    calibration_days: int,
    test_days: int,
    steps_per_day: int = 96,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series telemetry DataFrame chronologically into train, calibration, and test splits.

    Args:
        df: Input tabular telemetry DataFrame sorted by timestep.
        train_days: Number of days in clean training set.
        calibration_days: Number of days in clean calibration set.
        test_days: Number of days in test set.
        steps_per_day: Power-flow intervals per day (default 96 for 15-min resolution).

    Returns:
        Tuple of (train_df, calibration_df, test_df).

    Raises:
        ValueError if dataset length is insufficient or timestamps overlap.
    """
    total_days_needed = train_days + calibration_days + test_days
    max_timestep = df["timestep"].max()
    min_timestep = df["timestep"].min()
    total_timesteps = max_timestep - min_timestep + 1

    train_end_step = min_timestep + (train_days * steps_per_day) - 1
    calib_start_step = train_end_step + 1
    calib_end_step = calib_start_step + (calibration_days * steps_per_day) - 1
    test_start_step = calib_end_step + 1
    test_end_step = test_start_step + (test_days * steps_per_day) - 1

    if max_timestep < test_end_step:
        raise ValueError(
            f"Dataset has {total_timesteps} timesteps ({total_timesteps/steps_per_day:.1f} days), "
            f"but requested split requires {total_days_needed * steps_per_day} timesteps ({total_days_needed} days)."
        )

    train_df = df[(df["timestep"] >= min_timestep) & (df["timestep"] <= train_end_step)].copy()
    calib_df = df[(df["timestep"] >= calib_start_step) & (df["timestep"] <= calib_end_step)].copy()
    test_df = df[(df["timestep"] >= test_start_step) & (df["timestep"] <= test_end_step)].copy()

    # Integrity verification
    train_steps = set(train_df["timestep"].unique())
    calib_steps = set(calib_df["timestep"].unique())
    test_steps = set(test_df["timestep"].unique())

    if train_steps & calib_steps:
        raise ValueError("Data leakage detected: Train and Calibration timesteps overlap!")
    if train_steps & test_steps:
        raise ValueError("Data leakage detected: Train and Test timesteps overlap!")
    if calib_steps & test_steps:
        raise ValueError("Data leakage detected: Calibration and Test timesteps overlap!")

    # Verify train and calib contain only normal data
    if "is_attack" in train_df.columns and (train_df["is_attack"] > 0).any():
        raise ValueError("Integrity violation: Training set contains attack labels!")
    if "is_attack" in calib_df.columns and (calib_df["is_attack"] > 0).any():
        raise ValueError("Integrity violation: Calibration set contains attack labels!")

    return train_df, calib_df, test_df
