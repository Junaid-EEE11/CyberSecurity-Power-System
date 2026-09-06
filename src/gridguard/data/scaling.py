"""Feature standardization scaler fitted strictly on training data."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch


class FeatureScaler:
    """Standardizes feature columns using mean and std calculated strictly from training data."""

    def __init__(self, feature_cols: Optional[List[str]] = None):
        """Initialize scaler.

        Args:
            feature_cols: List of column names to scale.
        """
        self.feature_cols = feature_cols or ["v_mag_pu", "v_ang_rad", "p_inj_pu", "q_inj_pu"]
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler parameters (mean, std) on clean training data.

        Args:
            df_train: Training DataFrame.

        Returns:
            Fitted self.
        """
        for col in self.feature_cols:
            if col not in df_train.columns:
                raise ValueError(f"Feature column '{col}' not found in training DataFrame.")
            mean_val = float(df_train[col].mean())
            std_val = float(df_train[col].std())
            # Prevent zero standard deviation
            if std_val < 1e-8:
                std_val = 1.0
            self.means[col] = mean_val
            self.stds[col] = std_val

        self.fitted = True
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization to a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed copy of DataFrame with scaled columns.
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
        df_scaled = df.copy()
        for col in self.feature_cols:
            df_scaled[col] = (df[col] - self.means[col]) / self.stds[col]
        return df_scaled

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Inverse standardize a 2D numpy array [N, F]."""
        X_inv = np.zeros_like(X_scaled)
        for idx, col in enumerate(self.feature_cols):
            X_inv[:, idx] = X_scaled[:, idx] * self.stds[col] + self.means[col]
        return X_inv

    def transform_array(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Standardize a specific feature array."""
        col_name = self.feature_cols[feature_idx]
        return (X - self.means[col_name]) / self.stds[col_name]

    def inverse_transform_tensor(self, X_scaled: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Inverse standardize a PyTorch tensor for a specific feature."""
        col_name = self.feature_cols[feature_idx]
        mean = self.means[col_name]
        std = self.stds[col_name]
        return X_scaled * std + mean

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Export scaler statistics to dictionary for serialization."""
        return {
            "means": self.means,
            "stds": self.stds,
            "feature_cols": {col: idx for idx, col in enumerate(self.feature_cols)},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureScaler":
        """Load scaler from dictionary."""
        feature_cols = list(data.get("feature_cols", {}).keys()) or list(data["means"].keys())
        scaler = cls(feature_cols=feature_cols)
        scaler.means = data["means"]
        scaler.stds = data["stds"]
        scaler.fitted = True
        return scaler
