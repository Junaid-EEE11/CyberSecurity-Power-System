"""Comprehensive evaluation, metrics, localization, delay, and statistical testing."""

from gridguard.evaluation.detection import compute_detection_metrics, compute_recall_at_far
from gridguard.evaluation.localization import compute_localization_metrics
from gridguard.evaluation.delay import compute_detection_delay
from gridguard.evaluation.statistics import (
    bootstrap_confidence_interval,
    paired_wilcoxon_test,
    holm_bonferroni_correction,
)

__all__ = [
    "compute_detection_metrics",
    "compute_recall_at_far",
    "compute_localization_metrics",
    "compute_detection_delay",
    "bootstrap_confidence_interval",
    "paired_wilcoxon_test",
    "holm_bonferroni_correction",
]
