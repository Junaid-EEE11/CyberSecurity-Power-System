"""Split-conformal calibration for calibrated false-alarm rate control."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from gridguard.calibration.thresholds import (
    compute_quantile_threshold,
    compute_empirical_far,
)


class ConformalCalibrator:
    """Calibrates anomaly thresholds and computes empirical p-values from clean scores."""

    def __init__(self, alpha_values: Optional[List[float]] = None):
        """Initialize calibrator.

        Args:
            alpha_values: Target alpha levels (e.g. [0.01, 0.05]).
        """
        self.alpha_values = alpha_values or [0.01, 0.05]
        self.calib_scores: Optional[np.ndarray] = None
        self.thresholds: Dict[float, float] = {}
        self.fitted: bool = False

    def fit(self, calib_scores: np.ndarray) -> "ConformalCalibrator":
        """Compute conformal thresholds on clean calibration scores.

        Args:
            calib_scores: 1D array of scores evaluated strictly on clean normal calibration period.
        """
        self.calib_scores = np.sort(np.asarray(calib_scores, dtype=np.float64))
        for alpha in self.alpha_values:
            self.thresholds[alpha] = compute_quantile_threshold(self.calib_scores, alpha)
        self.fitted = True
        return self

    def predict(self, test_scores: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
        """Predict binary anomaly decisions at target alpha.

        Args:
            test_scores: 1D array of test anomaly scores.
            alpha: Desired significance level.

        Returns:
            Tuple of (binary_decisions [0 or 1], threshold_value).
        """
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted with clean calibration scores first.")
        if alpha not in self.thresholds:
            self.thresholds[alpha] = compute_quantile_threshold(self.calib_scores, alpha)

        threshold = self.thresholds[alpha]
        decisions = (test_scores > threshold).astype(np.int64)
        return decisions, threshold

    def compute_p_values(self, test_scores: np.ndarray) -> np.ndarray:
        """Compute empirical conformal p-values for test scores."""
        if not self.fitted or self.calib_scores is None:
            raise RuntimeError("Calibrator must be fitted before computing p-values.")

        n = len(self.calib_scores)
        # Vectorized empirical p-value computation
        # p_val = (1 + sum(s_calib >= s_test)) / (n + 1)
        # Using searchsorted on sorted calib_scores
        idx = np.searchsorted(self.calib_scores, test_scores, side="left")
        count_ge = n - idx
        p_values = (1.0 + count_ge) / (n + 1.0)
        return p_values

    def evaluate_clean_far(self, clean_test_scores: np.ndarray) -> Dict[float, Dict[str, float]]:
        """Evaluate empirical false-alarm rate on clean held-out test data for all target alphas."""
        results = {}
        for alpha in self.alpha_values:
            thresh = self.thresholds[alpha]
            emp_far = compute_empirical_far(clean_test_scores, thresh)
            results[alpha] = {
                "target_alpha": alpha,
                "threshold": thresh,
                "empirical_far": emp_far,
                "far_error": emp_far - alpha,
            }
        return results
