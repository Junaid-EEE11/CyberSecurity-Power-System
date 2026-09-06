"""Threshold computation and empirical false-alarm rate calculations."""

from typing import Tuple
import numpy as np


def compute_quantile_threshold(calib_scores: np.ndarray, alpha: float) -> float:
    """Compute split-conformal quantile threshold from clean calibration scores.

    Finite-sample conformal formula:
      index = ceil((n + 1) * (1 - alpha))
      threshold = sorted_scores[index - 1]

    Args:
        calib_scores: 1D array of anomaly scores from clean calibration data.
        alpha: Target significance level / false-alarm rate (e.g. 0.05).

    Returns:
        Calibrated decision threshold.
    """
    n = len(calib_scores)
    if n == 0:
        raise ValueError("Calibration score array is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

    sorted_scores = np.sort(calib_scores)
    # 1-indexed conformal rank
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    # Clip rank to available elements [1, n]
    rank = min(max(1, rank), n)
    threshold = float(sorted_scores[rank - 1])
    return threshold


def compute_empirical_far(clean_scores: np.ndarray, threshold: float) -> float:
    """Calculate empirical False-Alarm Rate (FAR) on clean held-out samples."""
    if len(clean_scores) == 0:
        return 0.0
    false_alarms = np.sum(clean_scores > threshold)
    return float(false_alarms / len(clean_scores))
