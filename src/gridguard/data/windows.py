"""Sliding temporal window extraction ensuring boundaries do not cross splits."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def create_sliding_windows(
    df: pd.DataFrame,
    window_length: int = 12,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert tabular time-series DataFrame into sliding window tensor arrays.

    Args:
        df: Input DataFrame containing sorted timesteps.
        window_length: Number of consecutive timesteps per window (default: 12).
        feature_cols: List of numerical feature columns.

    Returns:
        Tuple of:
          - X: [Num_Windows, Window_Length, Num_Nodes, Num_Features]
          - Y_target: [Num_Windows, Num_Nodes, Num_Features] (final timestep features)
          - is_attack: [Num_Windows] binary indicator for final timestep
          - is_compromised: [Num_Windows, Num_Nodes] binary localization mask for final timestep
          - target_timesteps: [Num_Windows] integer timestep of the reconstructed sample
    """
    if feature_cols is None:
        feature_cols = ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"]

    timesteps = sorted(df["timestep"].unique())
    num_timesteps = len(timesteps)
    num_nodes = len(df["node_id"].unique())
    num_features = len(feature_cols)

    if num_timesteps < window_length:
        raise ValueError(
            f"Insufficient timesteps ({num_timesteps}) for window length ({window_length})."
        )

    # Pivot feature tables to [Timestep, Node, Feature]
    # Ensure consistent sorting by timestep and node_id
    df_sorted = df.sort_values(by=["timestep", "node_id"])

    # Extract numpy blocks
    feat_matrix = df_sorted[feature_cols].to_numpy(dtype=np.float32)
    feat_3d = feat_matrix.reshape((num_timesteps, num_nodes, num_features))

    attack_matrix = df_sorted["is_attack"].to_numpy(dtype=np.int64).reshape((num_timesteps, num_nodes))
    attack_global = (attack_matrix.sum(axis=1) > 0).astype(np.int64)

    compromised_matrix = df_sorted["is_compromised"].to_numpy(dtype=np.int64).reshape((num_timesteps, num_nodes))

    num_windows = num_timesteps - window_length + 1
    X_windows = np.zeros((num_windows, window_length, num_nodes, num_features), dtype=np.float32)
    Y_targets = np.zeros((num_windows, num_nodes, num_features), dtype=np.float32)
    is_attack_windows = np.zeros(num_windows, dtype=np.int64)
    compromised_windows = np.zeros((num_windows, num_nodes), dtype=np.int64)
    target_timesteps = np.zeros(num_windows, dtype=np.int64)

    for i in range(num_windows):
        X_windows[i] = feat_3d[i : i + window_length]
        Y_targets[i] = feat_3d[i + window_length - 1]
        is_attack_windows[i] = attack_global[i + window_length - 1]
        compromised_windows[i] = compromised_matrix[i + window_length - 1]
        target_timesteps[i] = timesteps[i + window_length - 1]

    return X_windows, Y_targets, is_attack_windows, compromised_windows, target_timesteps
