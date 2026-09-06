"""Test split-conformal calibration quantile and p-value computations."""

import numpy as np
import pytest
from gridguard.calibration.conformal import ConformalCalibrator
from gridguard.calibration.thresholds import compute_quantile_threshold, compute_empirical_far


def test_conformal_quantile_hand_calculation():
    # 9 scores: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # n = 9. For alpha = 0.1:
    # rank = ceil((9 + 1) * (1 - 0.1)) = ceil(10 * 0.9) = ceil(9.0) = 9.
    # 9th score is 9.0
    cal_scores = np.array([5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0])
    thresh = compute_quantile_threshold(cal_scores, alpha=0.1)
    assert thresh == 9.0

    # For alpha = 0.2:
    # rank = ceil((9 + 1) * 0.8) = 8.
    # 8th score is 8.0
    thresh_02 = compute_quantile_threshold(cal_scores, alpha=0.2)
    assert thresh_02 == 8.0


def test_conformal_calibrator_predict_and_far():
    cal_scores = np.linspace(0.0, 1.0, 100)
    calibrator = ConformalCalibrator(alpha_values=[0.05])
    calibrator.fit(cal_scores)

    thresh = calibrator.thresholds[0.05]
    # Clean test scores from same distribution
    clean_test = np.linspace(0.0, 1.0, 100)
    emp_far = compute_empirical_far(clean_test, thresh)

    # Empirical FAR should be very close to nominal alpha 0.05
    assert abs(emp_far - 0.05) <= 0.02
