"""Test detection, localization, and delay metric calculations against toy numerical cases."""

import numpy as np
import pytest
from gridguard.evaluation.detection import compute_detection_metrics, compute_recall_at_far
from gridguard.evaluation.localization import compute_localization_metrics
from gridguard.evaluation.delay import compute_detection_delay
from gridguard.evaluation.statistics import bootstrap_confidence_interval, holm_bonferroni_correction


def test_detection_metrics_toy_case():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Perfect discriminator
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0])

    metrics = compute_detection_metrics(y_true, scores, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["far"] == 0.0


def test_localization_metrics_toy_case():
    y_node_true = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    node_scores = np.array([
        [0.1, 0.9, 0.2, 0.3],  # Node 1 highest -> Top-1 hit
        [0.8, 0.2, 0.1, 0.0],  # Node 0 highest -> Top-1 hit
    ])
    is_attack = np.array([1, 1])

    loc = compute_localization_metrics(y_node_true, node_scores, is_attack)
    assert loc["top1_hit"] == 1.0
    assert loc["top3_hit"] == 1.0


def test_detection_delay_toy_case():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0])
    # Attack triggers at timestep index 3 (delay = 1 step from index 2 onset)
    scores = np.array([0.1, 0.1, 0.2, 0.9, 0.9, 0.1, 0.1])
    timesteps = np.arange(len(y_true))

    delay = compute_detection_delay(y_true, scores, threshold=0.5, timesteps=timesteps, step_duration_minutes=15.0)
    assert delay["mean_delay_steps"] == 1.0
    assert delay["mean_delay_minutes"] == 15.0
    assert delay["detection_rate"] == 1.0


def test_holm_bonferroni_correction():
    p_vals = [0.01, 0.04, 0.03]
    corrected = holm_bonferroni_correction(p_vals, alpha=0.05)
    assert len(corrected) == 3
    # Smallest p-value (0.01) multiplied by m=3 -> 0.03 < 0.05 (significant)
    assert corrected[0][1] is True
