"""B0: Robust statistical baseline using Median and Median Absolute Deviation (MAD)."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class RobustStatisticalDetector:
    """Baseline B0: Standardizes features using training median and MAD to compute robust anomaly scores."""

    def __init__(self, name: str = "B0_Statistical_MAD"):
        self.name = name
        self.medians: Optional[np.ndarray] = None
        self.mads: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "RobustStatisticalDetector":
        """Fit median and MAD on clean training features.

        Args:
            X_train: Array of shape [N_samples, N_nodes, N_features] or [N_samples, Dim].

        Returns:
            Fitted self.
        """
        # Compute median across sample dimension
        self.medians = np.median(X_train, axis=0)
        # MAD = median(|X - median|) * 1.4826 (for normal consistency)
        abs_diff = np.abs(X_train - self.medians)
        raw_mad = np.median(abs_diff, axis=0)
        # Prevent division by zero
        self.mads = np.where(raw_mad < 1e-6, 1.0, raw_mad * 1.4826)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute node-level and global anomaly scores.

        Args:
            X: Array of shape [N_samples, N_nodes, N_features].

        Returns:
            Tuple of (global_scores [N_samples], node_scores [N_samples, N_nodes]).
        """
        if not self.fitted:
            raise RuntimeError("Detector must be fitted before scoring.")

        # Standardized z-scores per node and feature
        z = np.abs(X - self.medians) / self.mads  # [N_samples, N_nodes, N_features]
        node_scores = np.max(z, axis=-1)  # [N_samples, N_nodes]
        global_scores = np.max(node_scores, axis=-1)  # [N_samples]

        return global_scores, node_scores
